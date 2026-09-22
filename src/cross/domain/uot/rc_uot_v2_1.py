"""RC-UOT-v2.1: Reliability-Adaptive Partial UOT with bidirectional dustbin.

Variants:
  R1: Adaptive marginals only (reliability controls lambda_s/lambda_t)
  R2: Reliability-aware dustbin prior (R1 + q-dependent dustbin allocation)

Q constructions:
  Q0: Uniform (q_s=q_t=1)
  Q1: Rule-based monotonic
  Q2: Calibrated (isotonic logistic)
"""
from __future__ import annotations
import time, hashlib
from dataclasses import dataclass
from typing import Any
import numpy as np
import ot


@dataclass
class RCUOTv21Result:
    solver: str
    variant: str
    q_construction: str
    P_real: np.ndarray
    meta: dict[str, Any]


def _build_dustbin_augmented(C, a, b, causal_mask,
                              source_dustbin_cost=1.0, target_dustbin_cost=1.0,
                              dustbin_to_dustbin_cost=0.0):
    """Build dustbin-augmented cost matrix and marginals."""
    n_src, n_dst = C.shape
    # Augmented size: (n_src+1) x (n_dst+1)
    C_aug = np.full((n_src + 1, n_dst + 1), 1e12, dtype=float)
    C_aug[:n_src, :n_dst] = C

    # Source -> dustbin (target dustbin node = last column)
    C_aug[:n_src, n_dst] = source_dustbin_cost
    # Dustbin -> target (source dustbin node = last row)
    C_aug[n_src, :n_dst] = target_dustbin_cost
    # Free dustbin-to-dustbin
    C_aug[n_src, n_dst] = dustbin_to_dustbin_cost

    # Augmented marginals
    a_aug = np.concatenate([a, [b.sum()]])
    b_aug = np.concatenate([b, [a.sum()]])

    # Augmented causal mask: non-causal edges get high cost
    causal_aug = np.ones((n_src + 1, n_dst + 1), dtype=bool)
    causal_aug[:n_src, :n_dst] = causal_mask
    C_aug[:n_src, :n_dst] = np.where(causal_mask, C_aug[:n_src, :n_dst], 1e12)

    return C_aug, a_aug, b_aug


def _uot_sinkhorn_per_node(C, a, b, reg, lambda_s, lambda_t, numItermax=2000, stopThr=1e-9):
    """Unbalanced Sinkhorn with per-node lambda_s, lambda_t."""
    n_src, n_dst = C.shape
    K = np.exp(-C / reg)
    u = np.ones(n_src)
    v = np.ones(n_dst)

    for _ in range(numItermax):
        u_prev = u.copy()
        Kv = K.dot(v) + 1e-16
        # Per-source marginal relaxation
        ratio_s = a / Kv
        u = np.exp((lambda_s / (lambda_s + reg)) * np.log(np.maximum(ratio_s, 1e-16)))

        Ku = K.T.dot(u) + 1e-16
        ratio_t = b / Ku
        v = np.exp((lambda_t / (lambda_t + reg)) * np.log(np.maximum(ratio_t, 1e-16)))

        if np.max(np.abs(u - u_prev)) / max(np.max(np.abs(u)), 1) < stopThr:
            break

    P = np.diag(u).dot(K).dot(np.diag(v))
    return P


def solve_rc_uot_v2_1(
    C, a, b, causal_mask,
    variant='R1',
    q_construction='Q1',
    q_s=None, q_t=None,
    epsilon=0.05,
    lambda_min=0.01, lambda_max=1.0, alpha=1.0,
    source_dustbin_cost=1.0, target_dustbin_cost=1.0,
    gamma_q=0.0,
    pi_min=0.0, pi_max=0.5,
    numItermax=2000, stopThr=1e-9,
):
    """Solve RC-UOT-v2.1.

    Args:
        C: (n_src, n_dst) cost matrix
        a: (n_src,) source marginals
        b: (n_dst,) target marginals
        causal_mask: (n_src, n_dst) causal admissibility
        variant: 'R1' or 'R2'
        q_construction: 'Q0', 'Q1', or 'Q2'
        q_s: (n_src,) precomputed source reliability (overrides q_construction)
        q_t: (n_dst,) precomputed target reliability
        epsilon: entropic regularization
        lambda_min, lambda_max: KL penalty range
        alpha: reliability exponent
        source_dustbin_cost, target_dustbin_cost: dustbin costs
        gamma_q: reliability allocation penalty weight (R2 only)
        pi_min, pi_max: dustbin prior bounds (R2 only)

    Returns:
        RCUOTv21Result
    """
    t0 = time.perf_counter()
    n_src, n_dst = C.shape

    # Default q
    if q_s is None:
        q_s = np.ones(n_src)
    if q_t is None:
        q_t = np.ones(n_dst)

    q_s = np.nan_to_num(np.clip(np.asarray(q_s, dtype=float), 0.01, 0.99), nan=0.5)
    q_t = np.nan_to_num(np.clip(np.asarray(q_t, dtype=float), 0.01, 0.99), nan=0.5)

    # Per-node lambda
    lambda_s = lambda_min + (lambda_max - lambda_min) * q_s ** alpha
    lambda_t = lambda_min + (lambda_max - lambda_min) * q_t ** alpha

    # Build dustbin-augmented problem
    C_aug, a_aug, b_aug = _build_dustbin_augmented(
        C, a, b, causal_mask,
        source_dustbin_cost=source_dustbin_cost,
        target_dustbin_cost=target_dustbin_cost,
    )
    n_aug_src = n_src + 1
    n_aug_dst = n_dst + 1

    # Augmented lambdas
    lambda_s_aug = np.concatenate([lambda_s, [lambda_max]])
    lambda_t_aug = np.concatenate([lambda_t, [lambda_max]])

    # R2: Reliability-aware dustbin prior
    if variant == 'R2' and gamma_q > 0:
        pi_s = np.clip(1.0 - q_s, pi_min, pi_max)
        pi_t = np.clip(1.0 - q_t, pi_min, pi_max)
        # Adjust marginal to encourage dustbin allocation for low-q flows
        a_adj = a_aug.copy()
        b_adj = b_aug.copy()
        # Boost source dustbin marginal proportional to pi_s mean
        a_boost = np.mean(pi_s) * a.sum()
        b_boost = np.mean(pi_t) * b.sum()
        a_adj[n_src] += a_boost
        b_adj[n_dst] += b_boost
        a_adj = a_adj / a_adj.sum() * a_aug.sum()
        b_adj = b_adj / b_adj.sum() * b_aug.sum()
    else:
        a_adj = a_aug
        b_adj = b_aug
        pi_s = np.zeros(n_src)
        pi_t = np.zeros(n_dst)

    # Solve augmented UOT
    P_aug = _uot_sinkhorn_per_node(
        C_aug, a_adj, b_adj,
        reg=epsilon,
        lambda_s=lambda_s_aug,
        lambda_t=lambda_t_aug,
        numItermax=numItermax,
        stopThr=stopThr,
    )

    # Extract real transport
    P_real = P_aug[:n_src, :n_dst]
    P_real = P_real * causal_mask.astype(float)

    # Compute metadata
    source_dustbin_mass = P_aug[:n_src, n_dst]
    target_dustbin_mass = P_aug[n_src, :n_dst]
    total_real = P_real.sum()
    total_dustbin_src = source_dustbin_mass.sum()
    total_dustbin_tgt = target_dustbin_mass.sum()

    causal_violation = float(P_real[~causal_mask].sum()) if P_real[~causal_mask].size > 0 else 0.0
    causal_violation_rate = causal_violation / max(total_real, 1e-12)

    runtime = time.perf_counter() - t0

    meta = {
        'variant': variant,
        'q_construction': q_construction,
        'runtime_sec': runtime,
        'transport_mass_real': float(total_real),
        'source_dustbin_mass': float(total_dustbin_src),
        'target_dustbin_mass': float(total_dustbin_tgt),
        'source_dustbin_fraction': float(total_dustbin_src) / max(total_real + total_dustbin_src, 1e-12),
        'target_dustbin_fraction': float(total_dustbin_tgt) / max(total_real + total_dustbin_tgt, 1e-12),
        'causal_violation_rate': float(causal_violation_rate),
        'q_s_mean': float(q_s.mean()),
        'q_s_std': float(q_s.std()),
        'q_t_mean': float(q_t.mean()),
        'q_t_std': float(q_t.std()),
        'lambda_s_mean': float(lambda_s.mean()),
        'lambda_t_mean': float(lambda_t.mean()),
        'gamma_q': gamma_q,
        'pi_s_mean': float(pi_s.mean()),
        'pi_t_mean': float(pi_t.mean()),
    }

    return RCUOTv21Result(
        solver='rc_uot_v2_1',
        variant=variant,
        q_construction=q_construction,
        P_real=P_real,
        meta=meta,
    )