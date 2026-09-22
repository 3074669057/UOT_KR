"""Frozen support-aware solver wrappers (SOFTWARE-ONLY REPAIR).

Mathematical basis: ZERO_MASS_SUPPORT_REDUCTION_PROPOSITION.md. These wrappers
solve the FROZEN transport problems restricted to the strictly positive
marginal support and embed the result with exact zeros. No epsilon floor, no
pseudo-count, no marginal change, no cost change, no reg/reg_m change, no
missing-data-policy change. All-zero side returns the all-zero plan without
calling the solver (decoders then yield no supported correspondence /
abstention per the frozen rules).
"""
from __future__ import annotations

import numpy as np


def support_aware_uot(a: np.ndarray, b: np.ndarray, C: np.ndarray,
                      reg: float, reg_m: float) -> np.ndarray:
    import ot
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    Ip = np.where(a > 0)[0]
    Jp = np.where(b > 0)[0]
    if Ip.size == 0 or Jp.size == 0:
        return np.zeros_like(C, dtype=float)
    P = ot.sinkhorn_unbalanced(a[Ip], b[Jp], C[np.ix_(Ip, Jp)], reg, reg_m)
    out = np.zeros_like(C, dtype=float)
    out[np.ix_(Ip, Jp)] = P
    return out


def support_aware_bot(a: np.ndarray, b: np.ndarray, C: np.ndarray,
                      reg: float) -> np.ndarray:
    import ot
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    Ip = np.where(a > 0)[0]
    Jp = np.where(b > 0)[0]
    if Ip.size == 0 or Jp.size == 0:
        return np.zeros_like(C, dtype=float)
    P = ot.sinkhorn(a[Ip], b[Jp], C[np.ix_(Ip, Jp)], reg)
    out = np.zeros_like(C, dtype=float)
    out[np.ix_(Ip, Jp)] = P
    return out


def old_uot(a, b, C, reg, reg_m):
    import ot
    return ot.sinkhorn_unbalanced(a, b, C, reg, reg_m)


def old_bot(a, b, C, reg):
    import ot
    return ot.sinkhorn(a, b, C, reg)
