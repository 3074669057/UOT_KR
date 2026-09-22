"""Fixed q-aware baselines for RC-UOT-v2.2.
Hungarian with proper square dustbin augmentation.
Min-cost-flow using networkx.
"""
from __future__ import annotations
import numpy as np


def _make_dustbin_cost(q):
    """Map reliability q to dustbin cost. Lower q -> easier to abstain."""
    return 0.5 + 1.5 * (1.0 - np.asarray(q, dtype=float))


def solve_hungarian_dustbin_fixed(C, a, b, cm, q_s, q_t, dustbin_factor=1.0):
    """Hungarian with explicit dummy rows/columns for unmatched assignment.

    Creates square matrix of size (n_src + n_dst) with:
    - Real->Real: original costs (only admissible edges)
    - Real Src -> Target Dustbin: q-dependent cost
    - Source Dustbin -> Real Target: q-dependent cost
    - Dustbin <-> Dustbin: 0 (free pass-through)
    """
    from scipy.optimize import linear_sum_assignment

    n_src, n_dst = C.shape
    N = n_src + n_dst  # square size

    s_dc = float(np.mean(_make_dustbin_cost(q_s))) * dustbin_factor
    t_dc = float(np.mean(_make_dustbin_cost(q_t))) * dustbin_factor

    # Build square cost matrix
    C_sq = np.full((N, N), 1e12, dtype=float)

    # Real src -> real dst (only admissible)
    C_sq[:n_src, :n_dst] = np.where(cm, C, 1e12)

    # Real src -> target dustbin columns [n_dst, N)
    # Each src can go to any target dustbin slot
    C_sq[:n_src, n_dst:] = s_dc

    # Source dustbin rows [n_src, N) -> real dst
    C_sq[n_src:, :n_dst] = t_dc

    # Dustbin <-> dustbin (free)
    C_sq[n_src:, n_dst:] = 0.0

    row_ind, col_ind = linear_sum_assignment(C_sq)
    P = np.zeros((n_src, n_dst), dtype=float)
    for ri, cj in zip(row_ind, col_ind):
        if ri < n_src and cj < n_dst and cm[ri, cj]:
            P[ri, cj] = 1.0

    return P


def solve_min_cost_flow_fixed(C, a, b, cm, q_s, q_t, dustbin_factor=1.0):
    """Min-cost flow using networkx.

    Flow network:
      source -> src nodes -> dst nodes -> sink
      Plus dustbin bypass edges.

    Each src node gets 1 unit of supply.
    Each dst node gets 1 unit of demand.
    Dustbin edges absorb unmatched mass.
    """
    import networkx as nx

    n_src, n_dst = C.shape
    s_dc = float(np.mean(_make_dustbin_cost(q_s))) * dustbin_factor
    t_dc = float(np.mean(_make_dustbin_cost(q_t))) * dustbin_factor

    G = nx.DiGraph()

    # Nodes: SRC_0..SRC_{n_src-1}, DST_0..DST_{n_dst-1}, SOURCE, SINK
    SRC = [f"SRC_{i}" for i in range(n_src)]
    DST = [f"DST_{j}" for j in range(n_dst)]
    SOURCE = "SOURCE"
    SINK = "SINK"

    # Source -> SRC nodes: supply 1, cost 0
    for i in range(n_src):
        G.add_edge(SOURCE, SRC[i], capacity=1.0, weight=0.0)

    # SRC -> DST edges: only admissible, capacity 1
    for i in range(n_src):
        for j in range(n_dst):
            if cm[i, j]:
                G.add_edge(SRC[i], DST[j], capacity=1.0, weight=float(C[i, j]))

    # SRC -> SINK (source dustbin): q-dependent cost
    for i in range(n_src):
        G.add_edge(SRC[i], SINK, capacity=1.0, weight=s_dc)

    # DST -> SINK (target demand): cost 0
    for j in range(n_dst):
        G.add_edge(DST[j], SINK, capacity=1.0, weight=0.0)

    # Set demands
    G.nodes[SOURCE]["demand"] = -n_src
    G.nodes[SINK]["demand"] = n_src

    try:
        flow_dict = nx.min_cost_flow(G)
    except nx.NetworkXUnfeasible:
        return np.zeros((n_src, n_dst), dtype=float)

    P = np.zeros((n_src, n_dst), dtype=float)
    for i in range(n_src):
        for j in range(n_dst):
            if DST[j] in flow_dict.get(SRC[i], {}):
                f = flow_dict[SRC[i]][DST[j]]
                if f > 0:
                    P[i, j] = float(f)

    return P * cm.astype(float)


# Factory for all fixed q-aware baselines
def get_fixed_qaware_baselines():
    """Return dict of name -> solver function."""
    from cross.domain.uot.rc_uot_v2_1 import solve_rc_uot_v2_1
    import ot

    def _balanced_sinkhorn(C, a, b, cm, q_s, q_t, **_):
        n_src, n_dst = C.shape
        s_dc = float(np.mean(_make_dustbin_cost(q_s)))
        t_dc = float(np.mean(_make_dustbin_cost(q_t)))
        C_aug = np.full((n_src + 1, n_dst + 1), 1e12)
        C_aug[:n_src, :n_dst] = np.where(cm, C, 1e12)
        C_aug[:n_src, n_dst] = s_dc
        C_aug[n_src, :n_dst] = t_dc
        C_aug[n_src, n_dst] = 0.0
        a_aug = np.concatenate([a, [b.sum()]])
        b_aug = np.concatenate([b, [a.sum()]])
        an = a_aug / (a_aug.sum() + 1e-12)
        bn = b_aug / (b_aug.sum() + 1e-12)
        try:
            P_aug = ot.sinkhorn(an, bn, C_aug, reg=0.05, numItermax=2000, stopThr=1e-9)
            P_aug = np.asarray(P_aug, dtype=float)
        except Exception:
            P_aug = np.zeros((n_src + 1, n_dst + 1))
        return P_aug[:n_src, :n_dst] * cm.astype(float)

    return {
        "hungarian_qaware_fixed": solve_hungarian_dustbin_fixed,
        "min_cost_flow_fixed": solve_min_cost_flow_fixed,
        "balanced_sinkhorn_qaware": _balanced_sinkhorn,
    }
