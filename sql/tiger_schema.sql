-- Q-Watt analytics schema for Tiger Cloud (TimescaleDB). Executed block by block by qwatt/tiger.py.
-- Block header:  -- @@ <name> [optional] [fallback-for=<name>]
--   optional      : failure is tolerated (feature may be missing on the free plan)
--   fallback-for  : only runs if the named block failed
-- Source of truth is SQLite; this is the analytics layer. All DDL is idempotent.

-- @@ ext_toolkit optional
CREATE EXTENSION IF NOT EXISTS timescaledb_toolkit;

-- @@ power_samples
CREATE TABLE IF NOT EXISTS power_samples (
    ts              timestamptz NOT NULL,
    run_id          text,
    arm             text,
    phase           text,
    pkg_watts       double precision,
    pp0_watts       double precision,
    pp1_watts       double precision,
    pkg_joules_cum  double precision,
    sys_watts       double precision,
    source          text,
    PRIMARY KEY (ts)
);

-- @@ power_samples_hypertable
SELECT create_hypertable('power_samples', 'ts', chunk_time_interval => INTERVAL '1 day',
                         if_not_exists => TRUE, migrate_data => TRUE);

-- @@ power_samples_idx
CREATE INDEX IF NOT EXISTS power_samples_run_idx ON power_samples (run_id, ts);

-- @@ runs
CREATE TABLE IF NOT EXISTS runs (
    run_id text PRIMARY KEY, qid text, question text, gold text, arm text, mode text,
    started_at timestamptz, ended_at timestamptz, config_hash text, llm_backend text, model_id text,
    tokens_in integer, tokens_out integer, context_tokens integer, n_selected integer,
    selected_json text,
    ttft_s double precision, select_s double precision, prefill_s double precision,
    generate_s double precision, total_s double precision,
    idle_watts double precision, joules_gross double precision, joules_net double precision,
    joules_select double precision, joules_prefill double precision, joules_generate double precision,
    energy_source text,
    answer text, em double precision, f1 double precision, sf_recall double precision,
    referee_correct integer, referee_score double precision, referee_reason text, referee_model text,
    bench_id text
);

-- @@ qaoa_runs
CREATE TABLE IF NOT EXISTS qaoa_runs (
    run_id text PRIMARY KEY, qid text, backend text, n_qubits integer, n_edges integer,
    depth integer, gate_count integer, shots integer, gamma_json text, beta_json text,
    wall_s double precision, joules double precision, best_cost double precision,
    exact_cost double precision, approx_ratio double precision, amplification double precision,
    opt_in_shots integer, dispatch text, feasible_fallback integer, diag_json text
);

-- @@ phase_markers
CREATE TABLE IF NOT EXISTS phase_markers (
    run_id text NOT NULL, phase text NOT NULL, t0 timestamptz, t1 timestamptz,
    PRIMARY KEY (run_id, phase)
);

-- @@ power_1s_cagg optional
CREATE MATERIALIZED VIEW IF NOT EXISTS power_1s
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT time_bucket(INTERVAL '1 second', ts) AS bucket,
       run_id, arm,
       avg(pkg_watts) AS avg_watts,
       max(pkg_watts) AS max_watts,
       count(*)       AS n
FROM power_samples
WHERE run_id IS NOT NULL
GROUP BY 1, run_id, arm
WITH NO DATA;

-- @@ power_1s_policy optional
SELECT add_continuous_aggregate_policy('power_1s',
       start_offset => INTERVAL '3 days', end_offset => INTERVAL '1 second',
       schedule_interval => INTERVAL '1 minute', if_not_exists => TRUE);

-- @@ power_1s_view fallback-for=power_1s_cagg
CREATE OR REPLACE VIEW power_1s AS
SELECT time_bucket(INTERVAL '1 second', ts) AS bucket,
       run_id, arm,
       avg(pkg_watts) AS avg_watts,
       max(pkg_watts) AS max_watts,
       count(*)       AS n
FROM power_samples
WHERE run_id IS NOT NULL
GROUP BY 1, run_id, arm;

-- @@ scoreboard_view
CREATE OR REPLACE VIEW scoreboard AS
WITH per_arm AS (
    SELECT arm,
           count(*)                                              AS n,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY joules_net)   AS median_joules_net,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY joules_gross) AS median_joules_gross,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY total_s)      AS median_total_s,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY ttft_s)       AS median_ttft_s,
           avg(tokens_in)                                        AS mean_tokens_in,
           sum(joules_net)                                       AS sum_joules_net,
           sum(joules_gross)                                     AS sum_joules_gross,
           coalesce(sum(referee_correct), 0)                     AS n_correct,
           avg(em)                                               AS mean_em,
           avg(f1)                                               AS mean_f1,
           avg(sf_recall)                                        AS mean_sf_recall
    FROM runs
    WHERE arm IS NOT NULL AND mode = 'bench'
    GROUP BY arm
), j AS (
    SELECT *,
           sum_joules_net   / NULLIF(n_correct, 0) AS joules_per_correct,
           sum_joules_gross / NULLIF(n_correct, 0) AS joules_per_correct_gross
    FROM per_arm
)
SELECT j.*,
       100.0 * (1 - j.joules_per_correct / NULLIF(f.joules_per_correct, 0)) AS pct_saved_vs_full
FROM j LEFT JOIN j f ON f.arm = 'full';

-- @@ run_joules_sql_view
-- Joules per run by trapezoidal integration of the watt series (window functions),
-- plus the exact counter difference (max - min of the cumulative RAPL energy counter).
CREATE OR REPLACE VIEW run_joules_sql AS
SELECT run_id, arm,
       sum((pkg_watts + prev_w) / 2.0 * extract(epoch FROM (ts - prev_ts))) AS joules_trapezoid,
       max(pkg_joules_cum) - min(pkg_joules_cum)                              AS joules_counter,
       count(*)                                                               AS n_samples
FROM (
    SELECT run_id, arm, ts, pkg_watts, pkg_joules_cum,
           lag(pkg_watts) OVER w AS prev_w,
           lag(ts)        OVER w AS prev_ts
    FROM power_samples
    WHERE run_id IS NOT NULL
    WINDOW w AS (PARTITION BY run_id ORDER BY ts)
) s
WHERE prev_ts IS NOT NULL AND extract(epoch FROM (ts - prev_ts)) < 2.0
GROUP BY run_id, arm;

-- @@ run_joules_toolkit_view optional
CREATE OR REPLACE VIEW run_joules_toolkit AS
SELECT run_id, arm,
       integral(time_weight('Linear', ts, pkg_watts), 'second') AS joules_toolkit
FROM power_samples
WHERE run_id IS NOT NULL
GROUP BY run_id, arm;
