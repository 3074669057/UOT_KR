"""Shared frozen machinery for the real-anchor external validation.

DESIGN-ONLY package for the proposed corrected study (1->N fan-out primary). This
module is self-contained: it does NOT import the 301-305 holdout package and must
never read any confirmatory holdout artifact. All frozen values are transcriptions of
the frozen upstream identity (candidate spec sha256 0f360add...); nothing here is
tunable at execution time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Frozen constants (transcribed from the frozen candidate / pipeline identity) #
# --------------------------------------------------------------------------- #

FROZEN = {
    "candidate_spec_sha256": "0f360addc1f2a220299d75b4aa9cad1ee1e504e8097ae5bacfc0da40542cc401",
    "reg": 0.05,                 # entropic epsilon for both UOT and BOT
    "reg_m": 0.5,                # UOT marginal relaxation
    "lambda_risk": 0.25,
    "causal_violation_penalty": 5.0,
    "max_delay_sec": 21600.0,
    "decode_threshold": 1e-9,
    "k": 5,                      # D4 mutual top-5
    "cost_weights": {            # amount-free renormalized, sum to 1
        "time": 0.25, "route": 0.15, "risk": 0.15, "evidence": 0.05, "novelty": 0.05,
    },
    "boot_b": 4000,
    "boot_seed": 20260904,       # NEW seed; 20240101 belongs to the sealed holdout
    "uot_max_final_err": 1e-7,
    "bot_max_residual": 1e-6,
    "catastrophic_margin": -0.10,   # gate C lower bound for CONDITIONAL - THRESHOLD_MM
    "fp_fn_ratio": 3.0,             # gate D(i)/(ii)
    "misattr_margin": 0.05,         # gate D(iii) absolute margin
    "tau_grid_quantiles": [0.01, 0.02, 0.05, 0.1, 0.2],
    "tmm_calibration_frac": 0.25,   # chronologically earliest slice for tau selection
    "min_clusters_for_bootstrap": 3,  # degeneracy rule for the cluster bootstrap
}

# Canonical topology mapping (v2; legacy raw field values are data references only).
LEGACY_PATTERN_MAP = {
    "one_to_one": "1to1",
    "many_to_one": "1toN_fanout",   # verified inversion: src_deg>1, dst_deg==1
    "one_to_many": "Nto1_merge",    # verified inversion: src_deg==1, dst_deg>1
    "many_to_many": "NtoM",
}

FORBIDDEN_TOKENS = [
    "conditional_plan_holdout_results",
    "301", "302", "303", "304", "305",
]


def scan_forbidden(*texts: str) -> list[str]:
    """Return the forbidden tokens found in any supplied path/config string."""
    hits: list[str] = []
    for text in texts:
        for tok in FORBIDDEN_TOKENS:
            if tok in str(text):
                hits.append(tok)
    return sorted(set(hits))


# --------------------------------------------------------------------------- #
# Frozen decode rules (verbatim transcriptions of the candidate definitions)   #
# --------------------------------------------------------------------------- #

def rank_desc(S: np.ndarray, axis: int) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    if axis == 1:
        R = np.zeros_like(S, dtype=int)
        for i in range(S.shape[0]):
            order = np.lexsort((np.arange(S.shape[1]), -S[i]))
            R[i, order] = np.arange(1, S.shape[1] + 1)
        return R
    R = np.zeros_like(S, dtype=int)
    for j in range(S.shape[1]):
        order = np.lexsort((np.arange(S.shape[0]), -S[:, j]))
        R[order, j] = np.arange(1, S.shape[0] + 1)
    return R


def conditional_scores(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    S_row = np.zeros_like(P, dtype=float)
    S_col = np.zeros_like(P, dtype=float)
    S_row[:, c > 0] = P[:, c > 0] / c[None, c > 0]
    S_col[r > 0, :] = P[r > 0, :] / r[r > 0, None]
    return S_row, S_col


def mutual_top5(S: np.ndarray, k: int | None = None) -> set[tuple[int, int]]:
    k = int(FROZEN["k"] if k is None else k)
    rr = rank_desc(S, 1)
    cr = rank_desc(S, 0)
    out: set[tuple[int, int]] = set()
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if rr[i, j] <= k and cr[i, j] <= k:
                out.add((i, j))
    return out


def conditional_edges(P: np.ndarray, k: int | None = None) -> set[tuple[int, int]]:
    S_row, S_col = conditional_scores(P)
    k = int(FROZEN["k"] if k is None else k)
    rr = rank_desc(S_row, 1)
    cr = rank_desc(S_col, 0)
    out: set[tuple[int, int]] = set()
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if rr[i, j] <= k and cr[i, j] <= k:
                out.add((i, j))
    return out


def threshold_mm_edges(C: np.ndarray, tau: float) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j in zip(*np.where(C <= tau))}


def one_to_one_edges(C: np.ndarray, mode: str) -> set[tuple[int, int]]:
    """Per-source top-1. mode='connector': min cost; mode='abctracer': argmax of -C.
    Stable tie-break by destination index."""
    out: set[tuple[int, int]] = set()
    for i in range(C.shape[0]):
        row = np.asarray(C[i], dtype=float)
        if mode == "connector":
            j = int(np.lexsort((np.arange(row.size), row))[0])
        elif mode == "abctracer":
            j = int(np.lexsort((np.arange(row.size), -row))[0])
        else:
            raise ValueError(mode)
        out.add((i, j))
    return out


# --------------------------------------------------------------------------- #
# Component evaluation                                                        #
# --------------------------------------------------------------------------- #

def evaluate_components(
    gt: dict[str, dict],  # src -> {"dsts": [dst...], "t0": float}
    pred_edges: set[tuple[str, str]],
) -> dict[str, dict]:
    """Strict component recovery + edge-level metrics + abstention. Unit = component."""
    per: dict[str, dict] = {}
    gt_edges: set[tuple[str, str]] = set()
    pred_edges = set(pred_edges)  # dedupe
    for src, info in gt.items():
        dsts = set(info["dsts"])
        for d in dsts:
            gt_edges.add((src, d))
        e_s = {(s, d) for (s, d) in pred_edges if s == src}
        missing = dsts - {d for (_, d) in e_s}
        extra = {d for (_, d) in e_s} - dsts
        per[src] = {
            "gt_degree": len(dsts),
            "recovered": int(not missing and not extra),
            "partial_recall": (len(dsts) - len(missing)) / max(len(dsts), 1),
            "fp_edges": len(extra),
            "fn_edges": len(missing),
            "abstained": int(len(e_s) == 0),
            "t0": float(info["t0"]),
        }
    tp = len(gt_edges & pred_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    edge_precision = tp / max(tp + fp, 1)
    edge_recall = tp / max(tp + fn, 1)
    edge_f1 = 2 * edge_precision * edge_recall / max(edge_precision + edge_recall, 1e-12)
    return {
        "per_component": per,
        "edge_tp": tp, "edge_fp": fp, "edge_fn": fn,
        "edge_precision": edge_precision, "edge_recall": edge_recall, "edge_f1": edge_f1,
    }


# --------------------------------------------------------------------------- #
# Cluster bootstrap (paired per-component differences)                        #
# --------------------------------------------------------------------------- #

def cluster_bootstrap(
    per_component: dict[str, dict],
    method_a: str,
    method_b: str,
    values: dict[str, dict[str, float]],
    clusters: dict[str, int] | None = None,
    B: int | None = None,
    seed: int | None = None,
) -> dict:
    """Paired per-component difference bootstrap. clusters[component] = cluster id.

    v2 default cluster unit is PRIMARY ADDRESS (actor-level dependence); callers may
    pass an explicit clusters mapping. If clusters is None, the legacy time-block
    fallback is used (kept only for internal robustness, never for the real study).
    """
    B = int(FROZEN["boot_b"] if B is None else B)
    seed = int(FROZEN["boot_seed"] if seed is None else seed)
    keys = sorted(set(values[method_a]) & set(values[method_b]))
    d = np.array([values[method_a][s] - values[method_b][s] for s in keys], dtype=float)
    if d.size == 0:
        return {"point": float("nan"), "ci": [float("nan"), float("nan")], "n": 0,
                "degenerate": True, "not_evaluable": True}
    if clusters is None:
        t0 = np.array([per_component[s]["t0"] for s in keys], dtype=float)
        ids = np.floor(t0 / (7 * 86400.0)).astype(int)
        cluster_map = {s: int(i) for s, i in zip(keys, ids)}
    else:
        cluster_map = {s: int(clusters[s]) for s in keys}
    cluster_ids = [cluster_map[s] for s in keys]
    groups = {}
    for s, cid in cluster_map.items():
        groups.setdefault(cid, []).append(s)
    n_clusters = len(groups)
    point = float(d.mean())
    if n_clusters < FROZEN["min_clusters_for_bootstrap"]:
        return {"point": point, "ci": [float("nan"), float("nan")], "n": len(keys),
                "n_clusters": n_clusters, "degenerate": True, "not_evaluable": True,
                "seed": seed, "B": B,
                "reason": "fewer clusters than min_clusters_for_bootstrap"}
    idx_of = {s: k for k, s in enumerate(keys)}
    group_idx = [np.array([idx_of[s] for s in groups[cid]], dtype=int)
                 for cid in sorted(groups)]
    rng = np.random.RandomState(seed)
    means = np.empty(B, dtype=float)
    for b in range(B):
        picks = [group_idx[z] for z in rng.randint(0, len(group_idx), size=len(group_idx))]
        idx = np.concatenate(picks)
        means[b] = d[idx].mean()
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return {"point": point, "ci": [lo, hi], "n": len(keys), "n_clusters": n_clusters,
            "degenerate": False, "not_evaluable": False, "seed": seed, "B": B}


# --------------------------------------------------------------------------- #
# Hash gate                                                                   #
# --------------------------------------------------------------------------- #

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hash_manifest(manifest_path: Path, base: Path) -> list[str]:
    """Return list of mismatch errors (empty = PASS)."""
    errors: list[str] = []
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # noqa: BLE001
        return [f"manifest unreadable: {exc}"]
    for rel, expected in (data.get("files") or {}).items():
        p = base / rel
        if not p.is_file():
            errors.append(f"missing: {rel}")
            continue
        actual = sha256_file(p)
        if actual != expected:
            errors.append(f"hash mismatch: {rel}")
    return errors
