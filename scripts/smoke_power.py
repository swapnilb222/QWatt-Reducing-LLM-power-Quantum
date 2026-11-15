"""Smoke test: verify the RAPL PDH counter units empirically.

Samples \\Energy Meter(RAPL_Package0_PKG)\\Energy and \\Power at ~10 Hz for a few seconds idle,
then during a CPU burn; compares delta(Energy) with mean(Power)*dt to infer the units.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import win32pdh  # noqa: E402


def _burn(stop_t):
    x = 0
    while time.time() < stop_t:
        x += 1


class Pdh:
    def __init__(self):
        self.q = win32pdh.OpenQuery()
        self.h = {}
        for inst in ("RAPL_Package0_PKG", "RAPL_Package0_PP0", "RAPL_Package0_PP1"):
            for c in ("Power", "Energy"):
                path = f"\\Energy Meter({inst})\\{c}"
                self.h[(inst, c)] = win32pdh.AddCounter(self.q, path)
        win32pdh.CollectQueryData(self.q)

    def read(self):
        win32pdh.CollectQueryData(self.q)
        out = {}
        for k, h in self.h.items():
            _t, v = win32pdh.GetFormattedCounterValue(h, win32pdh.PDH_FMT_DOUBLE)
            out[k] = v
        return out


def phase(p, seconds, label):
    rows = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        rows.append((time.time(), p.read()))
        time.sleep(0.1)
    t0, r0 = rows[0]
    t1, r1 = rows[-1]
    k = ("RAPL_Package0_PKG", "Power")
    e = ("RAPL_Package0_PKG", "Energy")
    mean_p = sum(r[k] for _, r in rows) / len(rows)
    dE = r1[e] - r0[e]
    dt = t1 - t0
    print(f"[{label}] dt={dt:.2f}s samples={len(rows)} meanPower_raw={mean_p:.1f} "
          f"dEnergy_raw={dE:.1f}  ratio dE/(meanP*dt)={dE / (mean_p * dt):.5f}")
    return mean_p, dE, dt


def main():
    p = Pdh()
    time.sleep(0.5)
    print("raw first read:", {f"{a[9:]}/{b}": round(v, 1) for (a, b), v in p.read().items()})
    mp_i, dE_i, dt_i = phase(p, 4, "idle")
    n = max(1, mp.cpu_count())
    stop_t = time.time() + 6
    procs = [mp.Process(target=_burn, args=(stop_t,)) for _ in range(n)]
    for pr in procs:
        pr.start()
    mp_b, dE_b, dt_b = phase(p, 4, "burn")
    for pr in procs:
        pr.join()
    J = 3.6e-9  # Energy counter is in picowatt-hours: 1 raw = 1e-12 Wh = 3.6e-9 J
    for label, mp_, dE, dt in (("idle", mp_i, dE_i, dt_i), ("burn", mp_b, dE_b, dt_b)):
        e_w = dE * J / dt
        print(f"[{label}] Power counter = {mp_ / 1000:.2f} W (if mW) ; "
              f"dEnergy*3.6e-9/dt = {e_w:.2f} W  -> agreement {e_w / (mp_ / 1000):.3f}")
    print("VERDICT: Power is in milliwatts, Energy is in picowatt-hours (1 raw = 3.6e-9 J) "
          "if both agreement ratios are ~1.0 and burn >> idle.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
