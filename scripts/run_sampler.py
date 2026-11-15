"""Run the power sampler (separate process). Normally started by qwatt.power.ensure_sampler().

Usage: python scripts/run_sampler.py [--hz 4] [--seconds N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwatt import config, power  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=config.Settings().sampler_hz)
    ap.add_argument("--seconds", type=float, default=None, help="stop after N seconds (default: run forever)")
    a = ap.parse_args()
    (config.ROOT / "logs").mkdir(exist_ok=True)
    power.run_sampler(hz=a.hz, max_seconds=a.seconds)


if __name__ == "__main__":
    main()
