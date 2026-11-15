"""Smoke test: run a GHZ circuit on the Quantum Rings backend through the Qiskit bridge.

API used (quantumrings-toolkit-qiskit 0.2.0, QuantumRingsLib 0.12):
    provider = QuantumRingsLib.QuantumRingsProvider(token=QR_TOKEN, name=QR_ACCOUNT)
    backend  = quantumrings.toolkit.qiskit.QrBackendV2(provider, num_qubits=n)
    qc_t     = qiskit.transpile(qc, backend)
    counts   = backend.run(qc_t, shots=shots).result().get_counts()

Run:  .venv/Scripts/python scripts/smoke_qr.py [n_qubits] [shots]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qiskit import QuantumCircuit, transpile  # noqa: E402

from qwatt import config  # noqa: E402


def main(n: int = 4, shots: int = 1000) -> int:
    if not config.QR_TOKEN or not config.QR_ACCOUNT:
        print("FAIL: QR_TOKEN / QR_ACCOUNT missing from .env")
        return 1
    import QuantumRingsLib
    from quantumrings.toolkit.qiskit import QrBackendV2

    t0 = time.perf_counter()
    provider = QuantumRingsLib.QuantumRingsProvider(token=config.QR_TOKEN, name=config.QR_ACCOUNT)
    acct = provider.active_account()
    max_q = int(acct.get("max_qubits", 0)) if isinstance(acct, dict) else None
    backend = QrBackendV2(provider, num_qubits=n)
    t_conn = time.perf_counter() - t0
    print(f"backend={backend.name} max_qubits(account)={max_q} n={n} connect_s={t_conn:.2f}")

    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()

    t1 = time.perf_counter()
    qc_t = transpile(qc, backend)
    job = backend.run(qc_t, shots=shots)
    counts = job.result().get_counts()
    wall = time.perf_counter() - t1
    print(f"counts={dict(counts)} wall_s={wall:.3f}")

    ok = set(counts) <= {"0" * n, "1" * n} and sum(counts.values()) == shots
    print("PASS" if ok else "FAIL: unexpected outcomes for GHZ")
    return 0 if ok else 1


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    sys.exit(main(*args))
