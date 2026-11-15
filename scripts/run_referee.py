"""Grade all runs lacking a referee verdict (Gemini, batched, cached)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwatt import config, referee  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(config.DB_PATH))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    n = referee.grade_pending(a.db, limit=a.limit)
    print(f"graded {n} runs; gemini requests used: {referee.requests_used()}/{referee.MAX_REQUESTS}")


if __name__ == "__main__":
    main()
