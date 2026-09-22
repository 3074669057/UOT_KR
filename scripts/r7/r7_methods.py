"""R7 decoder catalogue.

Every method in one cell consumes the SAME amount-free cost matrix ``C`` and the SAME
frozen UOT plan ``P`` -- the plan is solved exactly once per cell and reused by every
decoder (specification section 41).

Design of the rule family (specification section 15)
---------------------------------------------------
A *rule* is a triple ``(ranking signal, budget rule, acceptance test)``.  The four
candidate rule families share one implementation of the mutual-top-k primitive
(specification section 16: reuse the verified deterministic ranking, never fork it):

  * ``R-const``      budget k in {2,3,4,5,6,8}, bilateral mutual top-k
  * ``R-quantile``   k = ceil(Q_q) of the FROZEN truncated degree distribution, clipped
                     to [2,8]; q in {0.75,0.90,0.95}; ONE k shared by all bridges
  * ``R-adaptive``   k_i^S = clip(ceil(alpha*r_i/median(r_positive)),1,8) and
                     k_j^T = clip(ceil(alpha*c_j/median(c_positive)),1,8); symmetric on
                     both sides, alpha in {2,3,4,5,6,8}, k_max = 8
  * ``R-threshold``  no k at all: bilateral relative-maximum rule
                     K_ij/rowmax_i >= theta AND K_ij/colmax_j >= theta (log domain)

The *ranking signal* is what distinguishes the method arms, holding the rule fixed:

  ``UOT_KR``          rank rows and columns by the direct forensic kernel logK = -C/eps
  ``RAW_UOT_PLAN``    rank by the raw transport plan P
  ``CONDITIONAL_UOT`` rank rows by P_ij/c_j and columns by P_ij/r_i (dual-cancelling)
  ``SUPPORT_PLUS_K``  rank by logK but restricted to the HARD transport support P > 1e-9

Ties are broken canonically: score descending, then ascending canonical index -- the
same ``np.lexsort((index, -score))`` primitive the frozen R5/R6 decoders used.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

from r7.r7_common import (DUAL_SOFTMAX_TAU_GRID, K_MAX, SUPPORT_THRESHOLD, TIE_TOL)

# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #

def log_kernel(C: np.ndarray, eps: float) -> np.ndarray:
    """``logK = -C/eps``.  Sorting by ``logK`` is exactly equivalent to sorting by
    ``K = exp(-C/eps)``; the log form is used internally for numerical stability."""
    return -np.asarray(C, dtype=float) / float(eps)


def rank_desc(S: np.ndarray, axis: int) -> np.ndarray:
    """Dense rank (1 = best) within rows (axis=1) or columns (axis=0).

    Canonical tie-break: equal scores are ordered by ascending canonical index, exactly
    the frozen ``np.lexsort((np.arange(n), -S))`` convention.
    """
    S = np.asarray(S, dtype=float)
    R = np.zeros_like(S, dtype=np.int64)
    if axis == 1:
        n = S.shape[1]
        idx = np.arange(n)
        for i in range(S.shape[0]):
            order = np.lexsort((idx, -S[i]))
            R[i, order] = np.arange(1, n + 1)
    else:
        n = S.shape[0]
        idx = np.arange(n)
        for j in range(S.shape[1]):
            order = np.lexsort((idx, -S[:, j]))
            R[order, j] = np.arange(1, n + 1)
    return R


def rank_desc_masked(S: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    """Dense rank computed ONLY over the masked candidates.

    Cells outside ``mask`` get rank ``INT_MAX``-like sentinel so they can never satisfy a
    top-k test.  This is the HARD support filter of specification section 23: support
    exclusion happens at the candidate-set level, never as a soft penalty score.
    """
    S = np.asarray(S, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    BIG = np.iinfo(np.int64).max // 4
    R = np.full(S.shape, BIG, dtype=np.int64)
    if axis == 1:
        for i in range(S.shape[0]):
            cols = np.flatnonzero(mask[i])
            if cols.size == 0:
                continue
            order = np.lexsort((cols, -S[i, cols]))
            R[i, cols[order]] = np.arange(1, cols.size + 1)
    else:
        for j in range(S.shape[1]):
            rows = np.flatnonzero(mask[:, j])
            if rows.size == 0:
                continue
            order = np.lexsort((rows, -S[rows, j]))
            R[rows[order], j] = np.arange(1, rows.size + 1)
    return R


def _edges_from_mask(mask: np.ndarray, sids: Sequence[str], tids: Sequence[str]
                     ) -> list[tuple[str, str]]:
    ii, jj = np.nonzero(mask)
    return [(str(sids[i]), str(tids[j])) for i, j in zip(ii.tolist(), jj.tolist())]


# --------------------------------------------------------------------------- #
# budget rules
# --------------------------------------------------------------------------- #

def adaptive_budgets(mass_rows: np.ndarray, mass_cols: np.ndarray, alpha: float,
                     k_max: int = K_MAX) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric source/target adaptive budgets (specification section 15).

    ``k_i^S = clip(ceil(alpha * r_i / median(r_positive)), 1, k_max)``
    ``k_j^T = clip(ceil(alpha * c_j / median(c_positive)), 1, k_max)``

    ``r``/``c`` are the realized masses of the frozen UOT plan.  Both sides are treated
    identically; a fan-out-only adaptation is explicitly NOT allowed.
    """
    def _one(m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, dtype=float).ravel()
        pos = m[m > 0]
        med = float(np.median(pos)) if pos.size else 0.0
        if med <= 0:
            return np.ones_like(m, dtype=np.int64)
        raw = np.ceil(float(alpha) * m / med)
        return np.clip(raw, 1, int(k_max)).astype(np.int64)

    return _one(mass_rows), _one(mass_cols)


def quantile_k(pmf_support: Sequence[int], pmf: Sequence[float], q: float,
               lo: int = 2, hi: int = K_MAX) -> int:
    """``k = clip(ceil(Q_q), lo, hi)`` from the FROZEN truncated empirical distribution."""
    support = np.asarray(pmf_support, dtype=float)
    p = np.asarray(pmf, dtype=float)
    p = p / p.sum()
    cdf = np.cumsum(p)
    idx = int(np.searchsorted(cdf, float(q), side="left"))
    idx = min(idx, len(support) - 1)
    return int(min(max(math.ceil(float(support[idx])), lo), hi))


# --------------------------------------------------------------------------- #
# edge builders
# --------------------------------------------------------------------------- #

def mutual_topk_edges(row_scores: np.ndarray, col_scores: np.ndarray,
                      sids: Sequence[str], tids: Sequence[str],
                      k_row: int | np.ndarray, k_col: int | np.ndarray,
                      *, support_mask: np.ndarray | None = None
                      ) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray]:
    """Bilateral mutual top-k.  ``k_row``/``k_col`` may be scalars or per-node arrays.

    When ``support_mask`` is given the ranks are computed over the support ONLY, so a
    non-support cell can never be selected (hard filter).
    """
    if support_mask is None:
        rr = rank_desc(row_scores, 1)
        cr = rank_desc(col_scores, 0)
        ok_mask = np.ones(row_scores.shape, dtype=bool)
    else:
        rr = rank_desc_masked(row_scores, support_mask, 1)
        cr = rank_desc_masked(col_scores, support_mask, 0)
        ok_mask = np.asarray(support_mask, dtype=bool)

    kr = (np.full(row_scores.shape[0], int(k_row), dtype=np.int64)
          if np.isscalar(k_row) else np.asarray(k_row, dtype=np.int64).ravel())
    kc = (np.full(row_scores.shape[1], int(k_col), dtype=np.int64)
          if np.isscalar(k_col) else np.asarray(k_col, dtype=np.int64).ravel())
    mask = (rr <= kr[:, None]) & (cr <= kc[None, :]) & ok_mask
    return _edges_from_mask(mask, sids, tids), rr, cr


def relative_threshold_edges(row_log_scores: np.ndarray, col_log_scores: np.ndarray,
                             sids: Sequence[str], tids: Sequence[str], theta: float,
                             *, support_mask: np.ndarray | None = None
                             ) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray]:
    """Bilateral relative-maximum rule (specification section 15, R-threshold).

    ``K_ij / rowmax_i >= theta`` and ``K_ij / colmax_j >= theta``, evaluated as a log
    difference to avoid underflow:  ``logK_ij - rowmax(logK)_i >= log(theta)``.

    A row or column whose maximum is not finite (empty candidate set) emits nothing; no
    edge is ever added to avoid an empty prediction.
    """
    R = np.asarray(row_log_scores, dtype=float)
    Cm = np.asarray(col_log_scores, dtype=float)
    if support_mask is not None:
        sup = np.asarray(support_mask, dtype=bool)
        R = np.where(sup, R, -np.inf)
        Cm = np.where(sup, Cm, -np.inf)
    rmax = np.max(R, axis=1, keepdims=True)
    cmax = np.max(Cm, axis=0, keepdims=True)
    log_theta = math.log(float(theta))
    with np.errstate(invalid="ignore"):
        ok_row = (R - rmax) >= log_theta - 1e-15
        ok_col = (Cm - cmax) >= log_theta - 1e-15
    finite = np.isfinite(rmax) & np.isfinite(cmax)
    mask = ok_row & ok_col & np.broadcast_to(finite, R.shape)
    if support_mask is not None:
        mask = mask & np.asarray(support_mask, dtype=bool)
    rr = rank_desc(R, 1)
    cr = rank_desc(Cm, 0)
    return _edges_from_mask(mask, sids, tids), rr, cr


# --------------------------------------------------------------------------- #
# method arms
# --------------------------------------------------------------------------- #

def conditional_scores(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frozen conditional-plan decoder scores: cancel the opposite dual.

    ``S_row(i,j) = P_ij / c_j`` (cancels the destination scaling, keeps source
    competition); ``S_col(i,j) = P_ij / r_i`` (cancels the source scaling).  Endpoints
    with zero realized mass have no support and rank last.  Plain float64, no tunable
    epsilon -- verbatim semantics of ``dev_candidate2.cp_common.conditional_scores``.
    """
    P = np.asarray(P, dtype=float)
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    S_row = np.zeros_like(P, dtype=float)
    S_col = np.zeros_like(P, dtype=float)
    pos_c = c > 0
    pos_r = r > 0
    S_row[:, pos_c] = P[:, pos_c] / c[None, pos_c]
    S_col[pos_r, :] = P[pos_r, :] / r[pos_r, None]
    return S_row, S_col


SIGNAL_UOT_KR = "logK"
SIGNAL_RAW = "P"
SIGNAL_CONDITIONAL = "conditional"
SIGNAL_SUPPORT = "logK_support"


def signal_matrices(signal: str, logK: np.ndarray, P: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return ``(row_scores, col_scores, support_mask_or_None)`` for one method arm."""
    if signal == SIGNAL_UOT_KR:
        return logK, logK, None
    if signal == SIGNAL_RAW:
        return P, P, None
    if signal == SIGNAL_CONDITIONAL:
        s_row, s_col = conditional_scores(P)
        return s_row, s_col, None
    if signal == SIGNAL_SUPPORT:
        return logK, logK, (P > SUPPORT_THRESHOLD)
    raise ValueError(f"unknown signal {signal}")


def apply_rule(rule: dict[str, Any], row_scores: np.ndarray, col_scores: np.ndarray,
               sids: Sequence[str], tids: Sequence[str],
               *, realized_rows: np.ndarray, realized_cols: np.ndarray,
               support_mask: np.ndarray | None = None) -> list[tuple[str, str]]:
    """Apply one frozen rule to one ranking signal."""
    fam = rule["family"]
    if fam in ("R-const", "R-quantile"):
        k = int(rule["k"])
        edges, _, _ = mutual_topk_edges(row_scores, col_scores, sids, tids, k, k,
                                        support_mask=support_mask)
        return edges
    if fam == "R-adaptive":
        kr, kc = adaptive_budgets(realized_rows, realized_cols, float(rule["alpha"]),
                                  int(rule.get("k_max", K_MAX)))
        edges, _, _ = mutual_topk_edges(row_scores, col_scores, sids, tids, kr, kc,
                                        support_mask=support_mask)
        return edges
    if fam == "R-threshold":
        edges, _, _ = relative_threshold_edges(row_scores, col_scores, sids, tids,
                                               float(rule["theta"]),
                                               support_mask=support_mask)
        return edges
    raise ValueError(f"unknown rule family {fam}")


def conditional_edges(P: np.ndarray, sids: Sequence[str], tids: Sequence[str],
                      k: int = 5) -> list[tuple[str, str]]:
    """Frozen ``dev_candidate2.cp_common.conditional_edges`` (mutual top-k on the
    dual-cancelling scores).  Kept for the R5/R6 reproduction anchors."""
    s_row, s_col = conditional_scores(P)
    edges, _, _ = mutual_topk_edges(s_row, s_col, sids, tids, k, k)
    return edges


def cost_d4_edges(C: np.ndarray, sids: Sequence[str], tids: Sequence[str],
                  k: int = 5) -> list[tuple[str, str]]:
    """Frozen ``diag.ctd_common.cost_d4_edges``: mutual top-k on ASCENDING cost, which
    is rank-identical to mutual top-k on ``K = exp(-C/eps)`` for every eps > 0."""
    n, m = np.asarray(C).shape
    R = np.zeros((n, m), dtype=np.int64)
    idx_m = np.arange(m)
    for i in range(n):
        order = np.lexsort((idx_m, C[i]))
        R[i, order] = np.arange(1, m + 1)
    Cc = np.zeros((n, m), dtype=np.int64)
    idx_n = np.arange(n)
    for j in range(m):
        order = np.lexsort((idx_n, C[:, j]))
        Cc[order, j] = np.arange(1, n + 1)
    mask = (R <= k) & (Cc <= k)
    return _edges_from_mask(mask, sids, tids)


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #

def hungarian_1to1(C: np.ndarray, sids: Sequence[str], tids: Sequence[str]
                   ) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """``HUNGARIAN_1TO1`` = cost-optimal one-to-one assignment baseline.

    Rectangular linear sum assignment on the SAME cost matrix ``C`` used by every other
    arm.  LABEL-FREE: it consumes only ``C`` -- no truth, no template map, no marginals.
    It is NOT an F1 performance upper bound for one-to-one methods; it is the best
    one-to-one assignment achievable at the same forensic cost.
    """
    from scipy.optimize import linear_sum_assignment

    C = np.asarray(C, dtype=float)
    ri, ci = linear_sum_assignment(C)
    edges = [(str(sids[i]), str(tids[j])) for i, j in zip(ri.tolist(), ci.tolist())]
    info = {
        "matrix_shape": [int(C.shape[0]), int(C.shape[1])],
        "assignment_cardinality": int(len(edges)),
        "total_selected_cost": float(C[ri, ci].sum()),
        "reads_labels": False,
        "method_name_in_paper": "cost-optimal one-to-one assignment baseline",
    }
    return edges, info


def threshold_mm_edges(C: np.ndarray, sids: Sequence[str], tids: Sequence[str],
                       cutoff: float) -> list[tuple[str, str]]:
    """Frozen Threshold-MM rule (``run_threshold_many_match.predict_edges``):
    edge iff ``C_ij <= cutoff``, decided independently per cell, so 0 / 1 / many
    matches per source are all admissible."""
    C = np.asarray(C, dtype=float)
    return _edges_from_mask(C <= float(cutoff), sids, tids)


def dual_softmax_edges(logK: np.ndarray, sids: Sequence[str], tids: Sequence[str],
                       tau: float = 0.0) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """``DUAL_SOFTMAX`` -- LoFTR-style dual-softmax with bidirectional acceptance.

    Complete R7 scheme (the historical R5B module was retired as mis-specified and was
    never executed; see ``R5C_PRIOR_ART_CORRECTION_NOTE.md``):

      ``S = row_softmax(logK)``      row-wise competition
      ``T = col_softmax(logK)``      column-wise competition
      ``D = S * T``                  dual-softmax confidence
      accept ``(i,j)`` iff ``j == argmax D(i,.)`` AND ``i == argmax D(.,j)``
      (bidirectional / mutual-nearest-neighbour acceptance, LoFTR Sec. 3.3)
      and the confidence gate ``D_ij >= tau``.

    Canonical tie handling: ``argmax`` is resolved by the lowest canonical index on both
    axes (numpy's documented ``argmax`` behaviour), which is the same index tie-break the
    other decoders use.  Label-free: consumes only the frozen kernel.
    """
    X = np.asarray(logK, dtype=float)
    S = _softmax(X, axis=1)
    T = _softmax(X, axis=0)
    D = S * T
    j_star = np.argmax(D, axis=1)                 # best column per row
    i_star = np.argmax(D, axis=0)                 # best row per column
    edges: list[tuple[str, str]] = []
    for i in range(D.shape[0]):
        j = int(j_star[i])
        if int(i_star[j]) != i:
            continue
        if float(D[i, j]) < float(tau) - TIE_TOL:
            continue
        edges.append((str(sids[i]), str(tids[j])))
    info = {
        "acceptance": "bidirectional mutual nearest neighbour on D = row_softmax * col_softmax",
        "tau": float(tau),
        "n_candidate_mutual_pairs": int(np.sum(i_star[j_star] == np.arange(D.shape[0]))),
    }
    return edges, info


def _softmax(X: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(X, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(X - m)
    s = e.sum(axis=axis, keepdims=True)
    return e / np.where(s > 0, s, 1.0)


# --------------------------------------------------------------------------- #
# diagnostic: label-informed one-to-one ceiling
# --------------------------------------------------------------------------- #

def oracle_1to1_ceiling(positive_edges: Iterable[tuple[str, str]]
                        ) -> dict[str, Any]:
    """``ORACLE_1TO1_CEILING`` -- label-informed semantic ceiling of the one-to-one
    output space (specification section 3.1).  DIAGNOSTIC ONLY.

    Maximum-cardinality matching on the ground-truth positive bipartite graph.  With
    ``T`` true edges and maximum non-conflicting true edges ``M``:

        precision = 1,  recall = M/T,  F1 = 2M / (T + M)

    This is NOT a deployable method, it never participates in H1/H2, it never enters
    Holm, and it can never be used to produce a UOT_KR prediction.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    edges = [(str(s), str(d)) for s, d in positive_edges]
    srcs = sorted({s for s, _ in edges})
    dsts = sorted({d for _, d in edges})
    T = len(set(edges))
    if T == 0:
        return {"T": 0, "M": 0, "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "label_informed": True, "deployable": False}
    si = {s: i for i, s in enumerate(srcs)}
    di = {d: j for j, d in enumerate(dsts)}
    rows = [si[s] for s, _ in edges]
    cols = [di[d] for _, d in edges]
    g = csr_matrix((np.ones(len(edges)), (rows, cols)), shape=(len(srcs), len(dsts)))
    match = maximum_bipartite_matching(g, perm_type="column")
    M = int(np.sum(match >= 0))
    f1 = 2.0 * M / (T + M) if (T + M) > 0 else float("nan")
    return {
        "T": int(T), "M": int(M),
        "precision": 1.0, "recall": float(M) / T, "f1": float(f1),
        "label_informed": True,
        "deployable": False,
        "role": "label-informed semantic ceiling, not a deployable baseline",
    }
