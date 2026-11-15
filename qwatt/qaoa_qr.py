"""QAOA circuit in Qiskit, executed on the Quantum Rings backend (QrBackendV2), decoded to bitstrings.

Quantum Rings API (quantumrings-toolkit-qiskit 0.2.0 + QuantumRingsLib 0.12):
    provider = QuantumRingsLib.QuantumRingsProvider(token=QR_TOKEN, name=QR_ACCOUNT)
    backend  = QrBackendV2(provider, num_qubits=n)          # 'scarlet_quantum_rings', all-to-all, native rzz
    qc_t     = transpile(qc, backend)
    counts   = backend.run(qc_t, shots=shots).result().get_counts()

Circuit (plan 6.5), normalized Ising coefficients a, b:
    H on all qubits; per layer l: RZ(2 g_l a_i) on i, RZZ(2 g_l b_ij) on each edge, RX(2 b_l) on i; measure all.
Qiskit RZ(t)=exp(-i t Z/2), RZZ(t)=exp(-i t ZZ/2), RX(t)=exp(-i t X/2), so this is exactly
exp(-i b_l sum X) exp(-i g_l H_C), the same unitary as qaoa_sim.statevector.

BIT ORDER: Qiskit count keys print qubit 0 on the RIGHT. `decode_key` is the one and only place that
reverses them; everything downstream uses strings where char i == qubit i == sentence i.
"""
from __future__ import annotations

import os
import time

import numpy as np
from qiskit import QuantumCircuit, transpile

from qwatt import config

_PROVIDER = None
_BACKENDS: dict[int, object] = {}


def decode_key(key: str) -> str:
    """Qiskit counts key (qubit 0 rightmost, maybe space-separated registers) -> char i == qubit i."""
    return key.replace(" ", "")[::-1]


def decode_counts(counts: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, v in counts.items():
        kk = decode_key(k)
        out[kk] = out.get(kk, 0) + int(v)
    return out


def get_backend(n: int):
    """Cached QrBackendV2 with n qubits (provider construction ~0.2 s, done once per process)."""
    global _PROVIDER
    if n not in _BACKENDS:
        import QuantumRingsLib
        from quantumrings.toolkit.qiskit import QrBackendV2

        if _PROVIDER is None:
            if not config.QR_TOKEN or not config.QR_ACCOUNT:
                raise RuntimeError("QR_TOKEN / QR_ACCOUNT missing in .env")
            _PROVIDER = QuantumRingsLib.QuantumRingsProvider(token=config.QR_TOKEN, name=config.QR_ACCOUNT)
        _BACKENDS[n] = QrBackendV2(_PROVIDER, num_qubits=n)
    return _BACKENDS[n]


def build_circuit(a: np.ndarray, b: np.ndarray, gammas, betas, tol: float = 1e-9) -> QuantumCircuit:
    n = len(a)
    qc = QuantumCircuit(n, name="qwatt_qaoa")
    qc.h(range(n))
    edges = [(i, j, b[i, j]) for i in range(n) for j in range(i + 1, n) if abs(b[i, j]) > tol]
    for g, bt in zip(gammas, betas):
        for i in range(n):
            if abs(a[i]) > tol:
                qc.rz(2 * g * a[i], i)
        for i, j, w in edges:
            qc.rzz(2 * g * w, i, j)
        qc.rx(2 * bt, range(n))
    qc.measure_all()
    return qc


def run_qaoa(a: np.ndarray, b: np.ndarray, gammas, betas, shots: int = 1024,
             performance: str | None = None) -> tuple[dict[str, int], dict]:
    """Run on Quantum Rings. Returns (decoded counts {bits: count}, circuit info)."""
    n = len(a)
    qc = build_circuit(a, b, gammas, betas)
    t0 = time.perf_counter()
    backend = get_backend(n)
    t1 = time.perf_counter()
    # No transpile(qc, backend) here: against the 200-qubit all-to-all Target it costs ~3 s (layout/routing) at
    # n=200, more than the simulation. QrBackendV2.run() itself transpiles to the QR basis gates (opt level 3).
    opts = {"shots": int(shots)}
    # Measured on the 265U (lead, 2:30 AM): max_threads=2 halves wall time (2.5 s -> 1.25 s) and cuts
    # circuit energy ~35% (75 J -> 48 J gross) vs the SDK default at 8192 shots. 0 = SDK default.
    threads = int(os.getenv("QWATT_QR_THREADS", "2"))
    if threads > 0:
        opts["max_threads"] = threads
    if performance:
        opts["performance"] = performance
    raw = backend.run(qc, **opts).result().get_counts()
    t2 = time.perf_counter()
    # same basis-gate transpile run() does internally (~0.04 s at n=200), only for the reported stats
    qc_t = transpile(qc, basis_gates=backend._qr_backend._basis_gates, optimization_level=3)
    # depth / gate_count describe the logical QAOA circuit (H, RZ, RZZ, RX) that we wrote; the transpiled
    # numbers (1-qubit gates merged into U) are reported alongside.
    ops = {k: int(v) for k, v in qc.count_ops().items()}
    ops_t = {k: int(v) for k, v in qc_t.count_ops().items()}
    no_meas = ("measure", "barrier")
    info = {
        "backend": str(backend.name), "n_qubits": n,
        "n_edges": int(sum(1 for i in range(n) for j in range(i + 1, n) if abs(b[i, j]) > 1e-9)),
        "depth": int(qc.remove_final_measurements(inplace=False).depth()),
        "gate_count": int(sum(v for k, v in ops.items() if k not in no_meas)),
        "ops": ops, "transpiled_depth": int(qc_t.remove_final_measurements(inplace=False).depth()),
        "transpiled_gate_count": int(sum(v for k, v in ops_t.items() if k not in no_meas)), "transpiled_ops": ops_t,
        "shots": int(shots), "max_threads": threads,
        "gamma": [float(g) for g in gammas], "beta": [float(x) for x in betas],
        "circuit_wall_s": t2 - t1, "backend_init_s": t1 - t0,
    }
    return decode_counts(raw), info
