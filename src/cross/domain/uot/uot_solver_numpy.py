"""Simplified UOT-style Sinkhorn without POT (for ablations / tests)."""
from __future__ import annotations

import numpy as np


def uot_sinkhorn(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    epsilon: float = 0.05,
    tau: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    """Return transport plan ``P`` of shape ``(len(a), len(b))``."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    c = np.asarray(c, dtype=float)
    a = a / (a.sum() + 1e-12)
    b = b / (b.sum() + 1e-12)

    k = np.exp(-c / max(float(epsilon), 1e-12))

    u = np.ones_like(a)
    v = np.ones_like(b)

    power = tau / (tau + max(float(epsilon), 1e-12))

    for _ in range(int(max_iter)):
        u_prev = u.copy()

        kv = k @ v + 1e-12
        u = (a / kv) ** power

        ktu = k.T @ u + 1e-12
        v = (b / ktu) ** power

        if float(np.linalg.norm(u - u_prev, ord=1)) < tol:
            break

    p = (u[:, None] * k) * v[None, :]
    return p
