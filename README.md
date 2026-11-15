# Q-Watt

**A quantum circuit decides which sentences of a document are worth the electricity.**

Local LLMs are increasingly handed whole documents (manuals, reports, articles). On a 15 W laptop, reading a
4,000–5,000-token document takes about 45 seconds of maxed-out CPU and about 265 J of net energy, for every
single question. Most of those sentences have nothing to do with the question.

Q-Watt puts **one qubit on every sentence of the whole document**, which is 116–200 qubits on the
**Quantum Rings SDK**, with no classical shortlist. A QAOA circuit chooses which sentences the model reads. The
energy price inside the Hamiltonian is measured on this laptop. A **safety net** keeps answers correct: if the
model cannot answer from the excerpt, Q-Watt re-reads the full document, and both reads are charged.

Built at Hack The Nest 2026 (Quantum Leap track, plus the Gemini API track).

---

## Results: 20 questions over 5 long Wikipedia documents (SQuAD), Gemma 4 E4B on CPU

Energy is the Intel RAPL CPU-package counter, net of an idle baseline measured before every question. The
quantum layer's own energy is charged to Q-Watt.

| | Full document | **Q-Watt (quantum + safety net)** |
|---|---|---|
| Tokens read per question | 4,373 | **524** |
| Energy per question (net) | 265 J | **61.5 J** |
| **Median energy saving** | | **83% (95% CI 81–84%)** |
| Time per question | 41 s | **10.7 s** |
| Answers correct (F1 ≥ 0.6) | 18 / 20 | **19 / 20** |
| Answer quality (SQuAD F1) | 0.89 | **0.90** |
| **Joules per correct answer** | 295 J | **65 J** |
| Safety net used | | 1 of 20 questions |

On this benchmark Q-Watt reads about one ninth of the document. It answers as well as reading the whole
document, using about a quarter of the energy and a quarter of the time.

### Why a quantum circuit, and why 200 qubits

- **The search space.** A 139-sentence document has 2^139 ≈ 10^41 possible excerpts, so brute force is out of
  the question. The Hamiltonian rewards relevance and coherent neighbouring sentences, penalizes repetition, and
  charges a measured energy price per token.
- **The circuit answers directly within the energy budget.** We raise the energy price inside the Hamiltonian
  until the circuit's typical measurement fits the token budget (predicted exactly from the p=1 marginals, then
  measured). On 100 test questions, **95% of QAOA shots fit the budget, against 0% of uniformly random
  selections**. Every selection comes straight from a measured shot, with no classical repair. The cost
  distributions of the circuit's shots and of random selections do not overlap.
- **The quantum layer costs little and grows slowly, while the savings grow linearly** (measured on this laptop,
  256 shots):

| Qubits (sentences) | Circuit time | Quantum-layer energy | Prefill energy saved |
|---|---|---|---|
| 16 | 0.20 s | 8.5 J | 17 J |
| 50 | 0.42 s | 11.1 J | 73 J |
| 100 | 0.64 s | 17.5 J | 176 J |
| 150 | 0.89 s | 22.6 J | 268 J |
| 200 | 1.36 s | 30.3 J | 381 J |

  The longer the document, the more the circuit pays for itself.
- **It runs fast because the Hamiltonian is local.** Each sentence couples only to its neighbours
  (|i − j| ≤ 2), so the circuit stays shallow (depth about 13) and Quantum Rings simulates 200 qubits in about
  a second. Shots are joules too: we measured the cost per shot and use 256.
- **Exact verification at small scale.** On 16-qubit instances we compare QAOA against brute force over all
  2^16 selections. Quantum Rings counts match the exact statevector up to shot noise. The shots contain the
  exact optimum about 9× more often than random sampling, with about 23× amplification of the best 1% of
  selections. See the dashboard's "Verified exactly at 16 qubits" view.

## How it works

```
document --> split into sentences (one qubit each) --> price them: relevance, coherence, repetition,
             measured joules per token
         --> QAOA p=1 on Quantum Rings (transferred angles, budget-calibrated energy price, 256 shots)
         --> best measured excerpt within the token budget --> Gemma 4 (LM Studio, CPU)
         --> model says NOT FOUND? --> safety net: re-read the full document (charged)
power sampler (separate process, Intel RAPL @ 4 Hz) --> SQLite --> per-phase joules --> dashboard
```

**Hamiltonian (QUBO, one binary variable per sentence):**
C(x) = −Σ rᵢ² xᵢ + μ Σ τᵢ xᵢ + λ Σ redᵢⱼ xᵢxⱼ − ν Σ cohᵢⱼ xᵢxⱼ, with couplings only for |i − j| ≤ 2.
The rent μ is tied to the calibrated joules per input token (0.059 J on this laptop) and raised until the
circuit's measurements fit the budget. It is mapped to an Ising Hamiltonian, RMS-normalized, and run as p=1
QAOA (H, RZ, RZZ, RX). Angles are trained once with an exact simulation on 16-sentence windows (γ = 1.02,
β = 2.39) and transferred to full documents, so there are zero optimizer iterations at runtime.

**Measurement.** Windows exposes Intel RAPL through the "Energy Meter" performance counters, with no admin
rights or driver. Joules per phase (select, prefill, generate, safety net) come from differencing the
cumulative energy counter at the phase markers.

**Fairness controls.** A random nonce opens every prompt, which defeats llama.cpp prefix-cache reuse. Arm order
is randomized per question, with a cooldown to idle and a fresh idle baseline before each question. Temperature
is 0, thinking is off, one prompt template serves both lanes, and every run stores a config hash.

## Run it

```bash
py -3.12 -m venv .venv && .venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env          # QR_TOKEN, QR_ACCOUNT, GEMINI_API_KEY
.venv\Scripts\python -m pytest -q

# LM Studio: Gemma 4 E4B, CPU, context length 8192, server on :1234
.venv\Scripts\python scripts\prepare_lc.py
.venv\Scripts\python scripts\run_bench_lc.py --bench-id my-run
.venv\Scripts\python -m streamlit run app\dashboard.py
```

| Path | What |
|---|---|
| `qwatt/longdoc.py` | SQuAD documents and one-qubit-per-sentence pools with local couplings |
| `qwatt/qubo.py` | QUBO, Ising mapping, budget-calibrated rent, budget decoding |
| `qwatt/qaoa_qr.py`, `qwatt/qaoa_sim.py` | Quantum Rings circuit execution; exact simulation, angle training, p=1 marginals |
| `qwatt/selectors.py` | Selection arms behind one interface |
| `qwatt/pipeline.py`, `qwatt/prompt.py`, `qwatt/llm.py`, `qwatt/bench.py` | One question through one arm, safety net, LM Studio client, randomized benchmark |
| `qwatt/power.py`, `qwatt/store.py` | RAPL sampler process, SQLite store, phase energy |
| `qwatt/referee.py` | Gemini referee (batched, cached) plus EM/F1 |
| `app/` | Streamlit dashboard: race, document-as-qubits map, why-quantum panel, scaling chart, scoreboard |
| `scripts/`, `tests/`, `results/` | Benchmarks, evaluations, measured results |

## Limitations

- The circuit runs on a simulator. The same Ising Hamiltonian runs unchanged on gate-model hardware.
- This is not a claim of quantum advantage. We claim a measured systems benefit, with the circuit's own energy
  on the ledger.
- The benchmark has 20 questions on one laptop, so the intervals are shown.

## References

- Singh, Vadlamani. *QUBO-Optimized Evidence Selection for Retrieval-Augmented QA with Unconventional Solvers.* arXiv:2607.12334 (2026).
- Shaydulin et al. *Parameter Transfer for QAOA of Weighted MaxCut.* ACM TQC (2023).
- Rajpurkar et al. *SQuAD* (CC BY-SA 4.0). Quantum Rings SDK and `quantumrings-toolkit-qiskit`.
