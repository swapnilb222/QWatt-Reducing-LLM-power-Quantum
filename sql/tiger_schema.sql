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

-- @@ derived_tables
-- Derived tables filled incrementally by tiger.refresh_derived() after each sync. Range-joining the (compressed)
-- sample hypertable per marker is expensive, so the SQL integrals are computed once per run and stored.
CREATE TABLE IF NOT EXISTS phase_energy_tbl (
    run_id text NOT NULL, phase text NOT NULL, t0 timestamptz, t1 timestamptz, dur_s double precision,
    n_samples integer, gross_j_tw double precision, gross_j_counter double precision,
    PRIMARY KEY (run_id, phase)
);
CREATE TABLE IF NOT EXISTS run_gaps_tbl (run_id text PRIMARY KEY, max_gap_s double precision);

-- @@ phase_energy_sql optional
-- Joules per (run, phase): power_samples joined to phase_markers by TIME RANGE (independent of the tags
-- written at sync time). gross_j_tw = time-weighted integral of watts (timescaledb_toolkit), gross_j_counter =
-- difference of the cumulative RAPL counter. Both are scaled by dur/span to correct for the samples that fall
-- just outside the phase edges (sampler period ~0.25 s).
CREATE OR REPLACE VIEW phase_energy_sql AS
SELECT run_id, phase, t0, t1, dur_s, n_samples,
       tw_j  * CASE WHEN span_s > 0 THEN dur_s / span_s ELSE 1 END AS gross_j_tw,
       ctr_j * CASE WHEN span_s > 0 THEN dur_s / span_s ELSE 1 END AS gross_j_counter
FROM (
    SELECT m.run_id, m.phase, m.t0, m.t1,
           extract(epoch FROM (m.t1 - m.t0))                                AS dur_s,
           extract(epoch FROM (max(s.ts) - min(s.ts)))                      AS span_s,
           integral(time_weight('Linear', s.ts, s.pkg_watts), 'second')     AS tw_j,
           max(s.pkg_joules_cum) - min(s.pkg_joules_cum)                    AS ctr_j,
           count(*)                                                         AS n_samples
    FROM phase_markers m
    JOIN power_samples s ON s.ts >= m.t0 AND s.ts <= m.t1
    WHERE m.t1 IS NOT NULL
    GROUP BY m.run_id, m.phase, m.t0, m.t1
) q;

-- @@ phase_ledger_view optional
-- Per (run, phase group): SQL joules (gross and NET of idle, same basis as runs.joules_*) next to the Python value.
DROP VIEW IF EXISTS phase_ledger CASCADE;
CREATE VIEW phase_ledger AS
WITH g AS (
    SELECT run_id,
           CASE WHEN phase LIKE 'escalate%' THEN 'escalate' ELSE phase END AS phase_group,
           sum(dur_s) AS dur_s, sum(gross_j_tw) AS sql_gross_j, sum(gross_j_counter) AS sql_counter_j
    FROM phase_energy_tbl GROUP BY 1, 2
)
SELECT g.run_id, r.arm, r.bench_id, r.qid, r.started_at, g.phase_group AS phase, g.dur_s,
       g.sql_gross_j, g.sql_counter_j,
       g.sql_gross_j - coalesce(r.idle_watts, 0) * g.dur_s   AS sql_net_j,
       CASE g.phase_group WHEN 'select' THEN r.joules_select WHEN 'prefill' THEN r.joules_prefill
                          WHEN 'generate' THEN r.joules_generate WHEN 'escalate' THEN r.joules_escalate END AS python_net_j
FROM g JOIN runs r USING (run_id);

-- @@ price_points_view optional
-- One row per run with a measured prefill: x = tokens_in, y = SQL net prefill joules (Python value if missing).
DROP VIEW IF EXISTS price_points CASCADE;
CREATE VIEW price_points AS
SELECT r.run_id, r.started_at, r.arm, r.tokens_in::double precision AS tokens_in,
       coalesce(p.sql_net_j, r.joules_prefill) AS prefill_j,
       CASE WHEN p.sql_net_j IS NOT NULL THEN 'sql' ELSE 'python' END AS y_source
FROM runs r LEFT JOIN phase_ledger p ON p.run_id = r.run_id AND p.phase = 'prefill'
WHERE r.tokens_in > 0 AND coalesce(r.escalated, 0) = 0 AND r.llm_backend <> 'fake'
  AND coalesce(p.sql_net_j, r.joules_prefill) > 0;

-- @@ energy_price_live_view optional
-- Live joules-per-input-token price: OLS of prefill joules on tokens_in over the 60 most recent runs.
DROP VIEW IF EXISTS energy_price_live CASCADE;
CREATE VIEW energy_price_live AS
SELECT regr_slope(prefill_j, tokens_in)     AS eps_in,
       regr_intercept(prefill_j, tokens_in) AS e0,
       regr_r2(prefill_j, tokens_in)        AS r2,
       count(*)                             AS n,
       max(started_at)                      AS as_of
FROM (SELECT * FROM price_points ORDER BY started_at DESC LIMIT 60) t;

-- @@ energy_price_drift_view optional
-- Same regression per 15-minute bucket, to show drift of the price over the session.
DROP VIEW IF EXISTS energy_price_drift CASCADE;
CREATE VIEW energy_price_drift AS
SELECT time_bucket(INTERVAL '15 minutes', started_at) AS bucket,
       regr_slope(prefill_j, tokens_in)     AS eps_in,
       regr_intercept(prefill_j, tokens_in) AS e0,
       regr_r2(prefill_j, tokens_in)        AS r2,
       sum(prefill_j) / NULLIF(sum(tokens_in), 0) AS j_per_token_avg,
       count(*)                             AS n
FROM price_points GROUP BY 1;

-- @@ run_integrity_view optional
-- Flags: idle baseline > 30% off the median of the previous 10 runs; a gap > 1.5 s between consecutive samples
-- inside the run; SQL (sum of phase integrals) vs Python gross joules differ by > 15%.
DROP VIEW IF EXISTS run_integrity CASCADE;
CREATE VIEW run_integrity AS
WITH base AS (
    SELECT run_id, arm, bench_id, started_at, ended_at, idle_watts, joules_gross FROM runs
    WHERE started_at IS NOT NULL AND ended_at IS NOT NULL
), gaps AS (
    SELECT run_id, max_gap_s FROM run_gaps_tbl
), sqlj AS (
    SELECT run_id, sum(gross_j_tw) AS sql_gross_j FROM phase_energy_tbl GROUP BY run_id
)
SELECT b.run_id, b.arm, b.bench_id, b.idle_watts, im.median_prev_idle,
       g.max_gap_s, sj.sql_gross_j, b.joules_gross AS python_gross_j,
       100.0 * (sj.sql_gross_j - b.joules_gross) / NULLIF(b.joules_gross, 0) AS sql_vs_python_pct,
       coalesce(abs(b.idle_watts - im.median_prev_idle) > 0.3 * im.median_prev_idle, false) AS flag_idle,
       coalesce(g.max_gap_s > 1.5, false)                                                   AS flag_gap,
       coalesce(abs(sj.sql_gross_j - b.joules_gross) > 0.15 * b.joules_gross, false)        AS flag_energy,
       (coalesce(abs(b.idle_watts - im.median_prev_idle) > 0.3 * im.median_prev_idle, false)
        OR coalesce(g.max_gap_s > 1.5, false)
        OR coalesce(abs(sj.sql_gross_j - b.joules_gross) > 0.15 * b.joules_gross, false))   AS flagged
FROM base b
LEFT JOIN gaps g USING (run_id)
LEFT JOIN sqlj sj USING (run_id)
LEFT JOIN LATERAL (
    SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY idle_watts) AS median_prev_idle FROM (
        SELECT r2.idle_watts FROM runs r2
        WHERE r2.started_at < b.started_at AND r2.idle_watts > 0 ORDER BY r2.started_at DESC LIMIT 10) q
) im ON true;

-- @@ scoreboard_view
ALTER TABLE runs ADD COLUMN IF NOT EXISTS escalated integer;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS joules_escalate double precision;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS answer_pruned text;
DROP VIEW IF EXISTS scoreboard;
-- Per benchmark batch and arm. "Correct" = Gemini referee verdict when present, else SQuAD/HotpotQA F1 >= 0.6.
-- Joule statistics use runs with positive net energy; percent saved is vs the batch's full-context arm.
CREATE VIEW scoreboard AS
WITH r AS (
    SELECT *, coalesce(referee_correct, CASE WHEN f1 >= 0.6 THEN 1 ELSE 0 END) AS correct
    FROM runs WHERE arm IS NOT NULL AND mode = 'bench' AND arm NOT IN ('topk', 'topk_lc')
      AND run_id NOT IN (SELECT run_id FROM run_integrity WHERE flagged)
), fl AS (
    SELECT r.bench_id, r.arm, count(*) FILTER (WHERE i.flagged) AS n_flagged,
           count(*) FILTER (WHERE i.flag_idle) AS n_flag_idle, count(*) FILTER (WHERE i.flag_gap) AS n_flag_gap,
           count(*) FILTER (WHERE i.flag_energy) AS n_flag_energy
    FROM runs r JOIN run_integrity i USING (run_id) WHERE r.mode = 'bench' GROUP BY 1, 2
), per_arm AS (
    SELECT bench_id, arm,
           count(*)                                                                         AS n,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY joules_net) FILTER (WHERE joules_net > 0) AS median_joules_net,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY joules_gross)                        AS median_joules_gross,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY total_s)                             AS median_total_s,
           avg(tokens_in)                                                                   AS mean_tokens_in,
           sum(joules_net) FILTER (WHERE joules_net > 0)                                    AS sum_joules_net,
           sum(correct) FILTER (WHERE joules_net > 0)                                       AS n_correct_valid,
           sum(correct)                                                                     AS n_correct,
           avg(f1)                                                                          AS mean_f1,
           sum(coalesce(escalated, 0))                                                      AS n_escalated
    FROM r GROUP BY bench_id, arm
), j AS (
    SELECT per_arm.*, coalesce(fl.n_flagged, 0) AS n_flagged, coalesce(fl.n_flag_idle, 0) AS n_flag_idle,
           coalesce(fl.n_flag_gap, 0) AS n_flag_gap, coalesce(fl.n_flag_energy, 0) AS n_flag_energy,
           sum_joules_net / NULLIF(n_correct_valid, 0) AS joules_per_correct
    FROM per_arm LEFT JOIN fl USING (bench_id, arm)
)
SELECT j.*,
       100.0 * (1 - j.median_joules_net / NULLIF(f.median_joules_net, 0))   AS pct_saved_median_vs_full,
       100.0 * (1 - j.joules_per_correct / NULLIF(f.joules_per_correct, 0)) AS pct_saved_per_correct_vs_full
FROM j LEFT JOIN j f ON f.bench_id = j.bench_id AND f.arm IN ('full', 'full_lc');

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

-- @@ compression optional
-- Columnstore: segment by run, order by time (newest first). Old chunks compress automatically.
ALTER TABLE power_samples SET (timescaledb.compress, timescaledb.compress_segmentby = 'run_id',
                               timescaledb.compress_orderby = 'ts DESC');

-- @@ compression_policy optional
SELECT add_compression_policy('power_samples', INTERVAL '1 hour', if_not_exists => TRUE);

-- @@ chunk_interval optional
SELECT set_chunk_time_interval('power_samples', INTERVAL '1 hour');

-- @@ power_1m_cagg optional
-- Hierarchical continuous aggregate: 1-minute buckets built on top of power_1s.
CREATE MATERIALIZED VIEW IF NOT EXISTS power_1m
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT time_bucket(INTERVAL '1 minute', bucket) AS bucket, run_id, arm,
       avg(avg_watts) AS avg_watts, max(max_watts) AS max_watts, sum(n) AS n
FROM power_1s GROUP BY 1, run_id, arm
WITH NO DATA;
