"""Connect to Tiger Cloud, create a tiny hypertable, insert, query, test cagg + schema, drop. Never prints the URL."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwatt import tiger  # noqa: E402


def main():
    if not tiger.configured():
        print("TIGER_DATABASE_URL is empty. Paste the service connection string into .env as "
              "TIGER_DATABASE_URL=postgres://... then rerun.")
        return 1
    conn = tiger.connect(timeout=10)
    if conn is None:
        print("could not connect (campus WiFi may block the Postgres port; try a phone hotspot)")
        return 2
    cur = conn.cursor()
    cur.execute("SELECT version()")
    print("postgres:", cur.fetchone()[0].split(",")[0])
    cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname LIKE 'timescaledb%'")
    print("extensions:", cur.fetchall())
    conn.execute("DROP TABLE IF EXISTS qwatt_smoke CASCADE")
    conn.execute("CREATE TABLE qwatt_smoke (ts timestamptz NOT NULL, w double precision)")
    conn.execute("SELECT create_hypertable('qwatt_smoke','ts')")
    conn.execute("INSERT INTO qwatt_smoke SELECT now() - (i||' seconds')::interval, 5 + i FROM generate_series(1,20) i")
    cur.execute("SELECT count(*), avg(w) FROM qwatt_smoke")
    print("rows/avg:", cur.fetchone())
    try:
        conn.execute("CREATE MATERIALIZED VIEW qwatt_smoke_cagg WITH (timescaledb.continuous) AS "
                     "SELECT time_bucket('1 second', ts) b, avg(w) a FROM qwatt_smoke GROUP BY 1 WITH NO DATA")
        print("continuous aggregates: supported")
    except Exception as e:  # noqa: BLE001
        print("continuous aggregates: NOT available ->", str(e).splitlines()[0][:100], "(fallback view will be used)")
    conn.execute("DROP TABLE IF EXISTS qwatt_smoke CASCADE")
    st = tiger.ensure_schema(conn)
    for k, v in st.items():
        print(f"  schema {k}: {v}")
    conn.close()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
