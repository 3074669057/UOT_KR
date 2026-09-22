"""Q5.7 probe: with C and k FIXED, does changing the risk-weighted source marginal a
change (a) the plan P, (b) the UOT_KR (logK) mutual-top-k edge set, (c) P-based decoders?"""
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(".").resolve()
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))
sys.path.insert(0, str(REPO / "scripts"))

EPS, REG_M = 0.05, 0.5
base = REPO / "out/r7_confirmatory_kernel_ranking_20260917/selection/generator/_cells"


def rank_desc(S, axis):
    S = np.asarray(S, float)
    R = np.zeros_like(S, dtype=np.int64)
    if axis == 1:
        n = S.shape[1]
        for i in range(S.shape[0]):
            o = np.lexsort((np.arange(n), -S[i]))
            R[i, o] = np.arange(1, n + 1)
    else:
        n = S.shape[0]
        for j in range(S.shape[1]):
            o = np.lexsort((np.arange(n), -S[:, j]))
            R[o, j] = np.arange(1, n + 1)
    return R


def mutual_topk(S, k=5):
    rr, cr = rank_desc(S, 1), rank_desc(S, 0)
    return set(zip(*np.where((rr <= k) & (cr <= k))))


for br in ["Celer", "Multi", "Poly"]:
    for d in sorted((base / br).glob("seed_*"))[:2]:
        cz = np.load(d / "cost.npz", allow_pickle=True)
        uz = np.load(d / "transport_uot.npz", allow_pickle=True)
        C = cz["C_primary"]
        a_rw, a0 = uz["a_rw"], uz["a0"]
        b = uz["b_ev"]
        logK = -C / EPS
        # solve with a_rw (as shipped) and with a0 (risk marginal disabled)
        import ot
        P1 = np.asarray(ot.unbalanced.sinkhorn_unbalanced(a_rw, b, C, reg=EPS, reg_m=REG_M,
                                                          numItermax=20000, stopThr=1e-11), float)
        P0 = np.asarray(ot.unbalanced.sinkhorn_unbalanced(a0, b, C, reg=EPS, reg_m=REG_M,
                                                          numItermax=20000, stopThr=1e-11), float)
        e_k = mutual_topk(logK)
        e_p1, e_p0 = mutual_topk(P1), mutual_topk(P0)

        def cond(P):
            r, c = P.sum(1), P.sum(0)
            Sr = np.zeros_like(P); Sc = np.zeros_like(P)
            pc, pr = c > 0, r > 0
            Sr[:, pc] = P[:, pc] / c[None, pc]
            Sc[pr, :] = P[pr, :] / r[pr, None]
            return mutual_topk(Sr), mutual_topk(Sc)
        cr1, cc1 = cond(P1)
        cr0, cc0 = cond(P0)
        print(f"--- {br}/{d.name}: n={C.shape[0]}x{C.shape[1]}  |a_rw-a0|max={np.abs(a_rw-a0).max():.3e}")
        print(f"    UOT_KR (logK=-C/eps) mutual-top5 edges: a_rw={len(e_k)} a0={len(e_k)}  "
              f"identical={e_k == e_k}   <-- risk marginal CANNOT matter here (C,k fixed)")
        print(f"    RAW PLAN P mutual-top5:  shipped={len(e_p1)} no-risk-marginal={len(e_p0)} "
              f"same={e_p1 == e_p0}  symmetric-diff={len(e_p1 ^ e_p0)}")
        print(f"    CONDITIONAL row:  same={cr1 == cr0} diff={len(cr1 ^ cr0)}   "
              f"col: same={cc1 == cc0} diff={len(cc1 ^ cc0)}")
        print(f"    plan diff: max|P1-P0|={np.abs(P1 - P0).max():.3e}  "
              f"rel={np.abs(P1 - P0).max() / max(P1.max(), 1e-30):.3e}")
