"""The decoder must recover a known single-qubit flip (plan 6.5 implementation trap)."""
import numpy as np
import pytest
from qiskit import QuantumCircuit

from qwatt import qaoa_qr, qaoa_sim, qubo


def test_decode_key_reverses_qiskit_order():
    # Qiskit prints qubit 0 rightmost: X on qubit 1 of 4 -> '0010'
    assert qaoa_qr.decode_key("0010") == "0100"
    assert qaoa_qr.decode_key("00 10") == "0100"


def _qr_available():
    from qwatt import config
    return bool(config.QR_TOKEN and config.QR_ACCOUNT)


@pytest.mark.skipif(not _qr_available(), reason="no Quantum Rings credentials")
@pytest.mark.parametrize("flip", [0, 2, 4])
def test_single_flip_on_quantum_rings(flip):
    n = 5
    qc = QuantumCircuit(n)
    qc.x(flip)
    qc.measure_all()
    backend = qaoa_qr.get_backend(n)
    counts = qaoa_qr.decode_counts(backend.run(qc, shots=50).result().get_counts())
    expected = "".join("1" if i == flip else "0" for i in range(n))
    assert counts == {expected: 50}
    assert qubo.str_to_bits(expected)[flip] == 1


def test_numpy_sim_matches_qiskit_statevector():
    """Same gates, same conventions: numpy QAOA probabilities == Qiskit Statevector of build_circuit."""
    from qiskit.quantum_info import Statevector

    rng = np.random.default_rng(0)
    n = 5
    a = rng.normal(size=n)
    b = np.triu(rng.normal(size=(n, n)) * (rng.random((n, n)) < 0.5), 1)
    b = b + b.T
    g, bt = [0.37, 0.81], [2.1, 0.6]
    qc = qaoa_qr.build_circuit(a, b, g, bt)
    qc.remove_final_measurements()
    p_qiskit = Statevector(qc).probabilities()      # index k: qubit i = bit i of k
    p_numpy = qaoa_sim.probabilities(qaoa_sim.ising_diagonal(a, b), g, bt)
    assert np.max(np.abs(p_qiskit - p_numpy)) < 1e-10
