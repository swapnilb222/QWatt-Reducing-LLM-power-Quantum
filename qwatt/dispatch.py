"""Dispatch gate (plan 6.8): only call the circuit when the instance has pairwise structure.

- closed_form_fits:      the whole pool fits the budget and keeping everything is trivially optimal
                         (every h_i <= 0 and every J_ij <= 0, so each kept sentence only lowers the cost).
- closed_form_separable: no nonzero couplings, so the problem separates: keep i exactly when h_i < 0,
                         used only when that choice fits the budget (otherwise it is a knapsack -> circuit).
- circuit:               otherwise.
"""
from __future__ import annotations

import numpy as np


def decide(h: np.ndarray, J: np.ndarray, t: np.ndarray, budget: float) -> tuple[str, np.ndarray | None]:
    """Returns (dispatch, x) where x is the closed-form solution or None for 'circuit'."""
    if t.sum() <= budget and (h <= 0).all() and (J <= 0).all():
        return "closed_form_fits", np.ones(len(h), dtype=np.uint8)
    if not np.any(np.abs(J) > 1e-12):
        x = (h < 0).astype(np.uint8)
        if x @ t <= budget:
            return "closed_form_separable", x
    return "circuit", None
