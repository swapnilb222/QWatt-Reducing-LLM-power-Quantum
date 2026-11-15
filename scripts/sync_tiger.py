"""Sync SQLite -> Tiger Data once, or repeatedly with --loop. No-ops if TIGER_DATABASE_URL is unset."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwatt import config, tiger  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--compare", action="store_true", help="print SQL-vs-Python joules after sync")
    a = ap.parse_args()
    if not tiger.configured():
        print("TIGER_DATABASE_URL not set; nothing to do.")
        return
    while True:
        res = tiger.sync_once(config.DB_PATH)
        print("synced:" if res is not None else "tiger unreachable", res or "")
        if a.compare:
            for r in (tiger.compare_joules(config.DB_PATH) or [])[:20]:
                print(r)
        if not a.loop:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
