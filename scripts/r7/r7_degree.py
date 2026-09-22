"""R7 Stage 0A -- empirical degree calibration from the frozen v4 / v5 audit windows.

Everything here reads *frozen audit artefacts only*.  No number is transcribed from
prose; every figure is recomputed from the canonical edge lists.

Outputs (under ``selection/degree_calibration/``)
-------------------------------------------------
``source_provenance.json``        path / sha256 / schema / row count / window identity
``degree_definition.json``        the frozen definition of "degree" (fan-out and fan-in)
``v4_degree_hist.csv``            per-window fan-out histogram
``v5_degree_hist.csv``            per-window fan-out histogram
``v4_fanin_hist.csv``             per-window fan-in histogram
``v5_fanin_hist.csv``             per-window fan-in histogram
``duplicate_units.json``          v4/v5 duplicate anchored-unit audit
``pooled_degree_hist_raw.csv``    pooled unique units, untruncated
``pooled_degree_hist_truncated.csv``  degrees 2..8, winsorised at 8
``pooled_fanin_hist_truncated.csv``   same for fan-in
``tail_report.json``              tail statistics + HIGH_TRUNCATION_TAIL
``degree_sampling_spec.json``     the frozen distribution the generator will sample
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

if __package__ in (None, ""):                      # allow `python scripts/r7/r7_degree.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r7.r7_common import (DEGREE_RANGE, DIR_DEGREE, K_MAX, REPO, V4_ROOT, V5_ROOT,
                          log, sha256_file, utc_now, write_json, write_text)

EDGES_NAME = "flow_edges_canonical_cumulative.json"
V4_UNITS_NAME = "primary_source_unit_list_v4.json"
V4_REPORT_NAME = "v4_final_report.json"
V5_MANIFEST_NAME = "V5_DATA_MANIFEST.json"
TAIL_ALERT = 0.10


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def _window_paths(tag: str) -> dict[str, Path]:
    root = V4_ROOT if tag == "v4" else V5_ROOT
    out = {"root": root, "edges": root / EDGES_NAME}
    if tag == "v4":
        out["units"] = root / V4_UNITS_NAME
        out["report"] = root / V4_REPORT_NAME
    else:
        out["manifest"] = root / V5_MANIFEST_NAME
    return out


def load_window(tag: str) -> dict[str, Any]:
    """One frozen audit window: canonical edges + provenance."""
    paths = _window_paths(tag)
    for key in ("edges",):
        if not paths[key].is_file():
            raise FileNotFoundError(
                f"frozen {tag} audit artefact missing: {paths[key]}. R7 Stage 0A cannot "
                f"calibrate the degree distribution without it.")
    edges = json.loads(paths["edges"].read_text(encoding="utf-8"))
    if not isinstance(edges, list) or not edges:
        raise ValueError(f"frozen {tag} canonical edge list is empty or malformed")
    prov: dict[str, Any] = {
        "window": tag,
        "root": str(paths["root"].relative_to(REPO)),
        "files": {},
        "row_count": len(edges),
    }
    for key, p in paths.items():
        if key == "root":
            continue
        if p.is_file():
            prov["files"][str(p.relative_to(REPO))] = {
                "sha256": sha256_file(p),
                "bytes": int(p.stat().st_size),
            }
    schema = sorted({k for r in edges[: min(len(edges), 5000)] for k in r.keys()})
    prov["record_schema"] = schema
    prov["required_identity_fields"] = [
        f for f in ("src_flow_id", "dst_flow_id", "support_anchors", "tier", "block")
        if f in schema]
    blocks = sorted({int(r.get("block", -1)) for r in edges})
    tiers = sorted({str(r.get("tier", "")) for r in edges})
    ctxs = sorted({str(r.get("ctx_src", "")) for r in edges})
    ts = [int(r["ts_min"]) for r in edges if r.get("ts_min") is not None]
    prov["window_identity"] = {
        "bridges_contexts": ctxs,
        "blocks": blocks,
        "n_blocks": len(blocks),
        "tiers": tiers,
        "ts_min": min(ts) if ts else None,
        "ts_max": max(ts) if ts else None,
        "ts_span_days": ((max(ts) - min(ts)) / 86400.0) if ts else None,
        "source_flow_prefix": sorted({str(r["src_flow_id"]).split("_")[0] for r in edges}),
    }
    # Unit counts straight from the canonical structure.
    src = {str(r["src_flow_id"]) for r in edges}
    dst = {str(r["dst_flow_id"]) for r in edges}
    prov["n_distinct_src_flow_ids"] = len(src)
    prov["n_distinct_dst_flow_ids"] = len(dst)
    prov["n_unique_canonical_edges"] = len({(str(r["src_flow_id"]), str(r["dst_flow_id"]))
                                            for r in edges})
    return {"tag": tag, "edges": edges, "provenance": prov}


# --------------------------------------------------------------------------- #
# degree extraction
# --------------------------------------------------------------------------- #

def canonical_unit_ids(edges: list[dict[str, Any]], side: str) -> dict[str, set[str]]:
    """canonical unit id -> set of distinct counterpart canonical ids.

    ``side='src'`` gives fan-out; ``side='dst'`` gives fan-in.  Identity uses the
    canonical flow ids that the frozen audit itself used (the same ids that appear as
    ``src_flow_id`` / ``dst_flow_id`` in the canonical edge list, which are themselves
    constructed from the canonical ``support_anchors`` transaction hashes).
    """
    if side == "src":
        a, b = "src_flow_id", "dst_flow_id"
    elif side == "dst":
        a, b = "dst_flow_id", "src_flow_id"
    else:
        raise ValueError(side)
    out: dict[str, set[str]] = {}
    for r in edges:
        out.setdefault(str(r[a]), set()).add(str(r[b]))
    return out


def canonical_anchor_keys(edges: list[dict[str, Any]]) -> dict[str, frozenset[str]]:
    """canonical src unit id -> frozenset of canonical ``support_anchors``.

    These tx hashes are the real-world identity of the anchoring events and are stable
    across windows, which is what the v4/v5 duplicate-unit check needs.
    """
    out: dict[str, set[str]] = {}
    for r in edges:
        out.setdefault(str(r["src_flow_id"]), set()).update(
            str(x) for x in (r.get("support_anchors") or []))
    return {k: frozenset(v) for k, v in out.items()}


def degree_hist(units: dict[str, set[str]], *, min_degree: int = 1,
                cap: int | None = None) -> Counter:
    h: Counter = Counter()
    for _, cs in units.items():
        d = len(cs)
        if d < min_degree:
            continue
        h[min(d, cap) if cap else d] += 1
    return h


def hist_rows(hist: Counter, *, label: str, total_denom: int | None = None) -> list[dict[str, Any]]:
    tot = int(total_denom if total_denom is not None else sum(hist.values()))
    return [{"distribution": label, "degree": int(d), "count": int(hist[d]),
             "probability": (hist[d] / tot) if tot else float("nan")}
            for d in sorted(hist)]


def write_hist_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["distribution", "degree", "count", "probability"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def quantiles(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {k: float("nan") for k in ("median", "q75", "q90", "q95", "q99")}
    return {k: float(np.percentile(v, p)) for k, p in
            (("median", 50), ("q75", 75), ("q90", 90), ("q95", 95), ("q99", 99))}


# --------------------------------------------------------------------------- #
# Stage 0A driver
# --------------------------------------------------------------------------- #

def calibrate() -> dict[str, Any]:
    DIR_DEGREE.mkdir(parents=True, exist_ok=True)

    windows = {t: load_window(t) for t in ("v4", "v5")}

    provenance = {
        "stage": "0A",
        "generated_at_utc": utc_now(),
        "purpose": ("Lock the empirical degree distribution used by the R7 generator to "
                    "the real frozen audit windows (paper section 4.6), recomputed from the "
                    "canonical audit artefacts rather than transcribed from prose."),
        "windows": {t: w["provenance"] for t, w in windows.items()},
        "definition_of_contamination_for_seeds": (
            "unrelated to this stage; see 00_preflight/seed_freshness_audit.json"),
    }
    log(f"[0A] provenance locked: v4 edges={windows['v4']['provenance']['row_count']} "
        f"v5 edges={windows['v5']['provenance']['row_count']}")

    # ---- per-window fan-out / fan-in ------------------------------------- #
    per_window: dict[str, Any] = {}
    for t, w in windows.items():
        fo = canonical_unit_ids(w["edges"], "src")
        fi = canonical_unit_ids(w["edges"], "dst")
        fo_hist_all = degree_hist(fo, min_degree=1)
        fi_hist_all = degree_hist(fi, min_degree=1)
        fo_hist_ge2 = degree_hist(fo, min_degree=2)
        fi_hist_ge2 = degree_hist(fi, min_degree=2)
        write_hist_csv(DIR_DEGREE / f"{t}_degree_hist.csv",
                       hist_rows(fo_hist_ge2, label=f"{t}_fanout_ge2"))
        write_hist_csv(DIR_DEGREE / f"{t}_fanin_hist.csv",
                       hist_rows(fi_hist_ge2, label=f"{t}_fanin_ge2"))
        per_window[t] = {
            "fanout_units_all": len(fo),
            "fanout_units_ge2": int(sum(fo_hist_ge2.values())),
            "fanin_units_all": len(fi),
            "fanin_units_ge2": int(sum(fi_hist_ge2.values())),
            "fanout_hist_all": {int(k): int(v) for k, v in sorted(fo_hist_all.items())},
            "fanout_hist_ge2": {int(k): int(v) for k, v in sorted(fo_hist_ge2.items())},
            "fanin_hist_all": {int(k): int(v) for k, v in sorted(fi_hist_all.items())},
            "fanin_hist_ge2": {int(k): int(v) for k, v in sorted(fi_hist_ge2.items())},
            "fanout_max": int(max(fo_hist_all)) if fo_hist_all else 0,
            "fanin_max": int(max(fi_hist_all)) if fi_hist_all else 0,
            "_fanout": fo, "_fanin": fi,
        }
        provenance["windows"][t]["window_statistics"] = {
            "canonical_edges": w["provenance"]["row_count"],
            "distinct_src_flow_ids": w["provenance"]["n_distinct_src_flow_ids"],
            "distinct_dst_flow_ids": w["provenance"]["n_distinct_dst_flow_ids"],
            "fanout_units_all": per_window[t]["fanout_units_all"],
            "fanout_units_ge2": per_window[t]["fanout_units_ge2"],
            "fanin_units_all": per_window[t]["fanin_units_all"],
            "fanin_units_ge2": per_window[t]["fanin_units_ge2"],
            "fanout_max": per_window[t]["fanout_max"],
            "fanin_max": per_window[t]["fanin_max"],
            "fanout_hist_ge2": per_window[t]["fanout_hist_ge2"],
            "fanin_hist_ge2": per_window[t]["fanin_hist_ge2"],
        }
    write_json(DIR_DEGREE / "source_provenance.json", provenance)

    # ---- v4/v5 duplicate anchored units ---------------------------------- #
    a4 = canonical_anchor_keys(windows["v4"]["edges"])
    a5 = canonical_anchor_keys(windows["v5"]["edges"])
    tx4 = {x for s in a4.values() for x in s}
    tx5 = {x for s in a5.values() for x in s}
    shared_tx = tx4 & tx5
    dup_units_v4 = sorted({u for u, s in a4.items() if s & shared_tx})
    dup_units_v5 = sorted({u for u, s in a5.items() if s & shared_tx})
    duplicate = {
        "canonical_unit_id_definition": (
            "The canonical identity of a protocol-anchored source unit is the set of its "
            "canonical support transaction hashes (``support_anchors`` in the frozen "
            "canonical edge list). Window-local ids (``v4_eth_...`` / ``v5_eth_...``) are "
            "NOT comparable across windows; the anchor hashes are."),
        "v4_anchors_total": len(tx4),
        "v5_anchors_total": len(tx5),
        "shared_anchor_count": len(shared_tx),
        "shared_anchor_examples": sorted(shared_tx)[:10],
        "duplicate_units_v4": len(dup_units_v4),
        "duplicate_units_v5": len(dup_units_v5),
        "duplicate_units_v4_examples": dup_units_v4[:10],
        "duplicate_units_v5_examples": dup_units_v5[:10],
        "dedup_applied": bool(shared_tx),
        "dedup_rule": ("drop the v5 copy of any source unit whose canonical anchor set "
                       "intersects the v4 anchor set; keep the v4 copy"),
    }
    write_json(DIR_DEGREE / "duplicate_units.json", duplicate)
    log(f"[0A] duplicate anchored units: shared anchors={len(shared_tx)} "
        f"v4 dups={len(dup_units_v4)} v5 dups={len(dup_units_v5)}")

    # ---- pooled unique units --------------------------------------------- #
    dropped = set(dup_units_v5) if shared_tx else set()
    pooled_fo: dict[str, set[str]] = {}
    pooled_fi: dict[str, set[str]] = {}
    for t in ("v4", "v5"):
        for u, cs in per_window[t]["_fanout"].items():
            if t == "v5" and u in dropped:
                continue
            pooled_fo[u] = cs
        for u, cs in per_window[t]["_fanin"].items():
            if t == "v5" and u in dropped:
                continue
            pooled_fi[u] = cs

    pooled_fo_ge2 = degree_hist(pooled_fo, min_degree=2)
    pooled_fi_ge2 = degree_hist(pooled_fi, min_degree=2)
    write_hist_csv(DIR_DEGREE / "pooled_degree_hist_raw.csv",
                   hist_rows(pooled_fo_ge2, label="pooled_fanout_ge2_raw"))
    write_hist_csv(DIR_DEGREE / "pooled_fanin_hist_raw.csv",
                   hist_rows(pooled_fi_ge2, label="pooled_fanin_ge2_raw"))

    # ---- truncation (winsorise at K_MAX, keep degree >= 2) ---------------- #
    def truncate(units: dict[str, set[str]]) -> tuple[Counter, dict[str, Any]]:
        degs = np.array([len(c) for c in units.values() if len(c) >= DEGREE_RANGE[0]],
                        dtype=int)
        if degs.size == 0:
            raise ValueError("no units with degree >= 2 in the pooled audit data")
        capped = np.minimum(degs, K_MAX)
        hist = Counter(capped.tolist())
        tot = int(capped.size)
        report = {
            "n_units_degree_ge_2": tot,
            "n_units_dropped_degree_lt_2": int(len(units) - tot),
            "p_degree_gt_cap": float(np.mean(degs > K_MAX)),
            "n_degree_gt_cap": int(np.sum(degs > K_MAX)),
            "observed_max_degree": int(degs.max()),
            "counts_2_to_cap": {int(d): int(hist[d]) for d in range(DEGREE_RANGE[0], K_MAX + 1)},
            "probs_2_to_cap": {int(d): float(hist[d] / tot)
                               for d in range(DEGREE_RANGE[0], K_MAX + 1)},
            **quantiles(degs),
        }
        return hist, report

    fo_hist, fo_tail = truncate(pooled_fo)
    fi_hist, fi_tail = truncate(pooled_fi)

    write_hist_csv(DIR_DEGREE / "pooled_degree_hist_truncated.csv",
                   hist_rows(fo_hist, label="pooled_fanout_ge2_winsor8"))
    write_hist_csv(DIR_DEGREE / "pooled_fanin_hist_truncated.csv",
                   hist_rows(fi_hist, label="pooled_fanin_ge2_winsor8"))

    high_tail = bool(max(fo_tail["p_degree_gt_cap"], fi_tail["p_degree_gt_cap"]) > TAIL_ALERT)
    tail_report = {
        "stage": "0A",
        "generated_at_utc": utc_now(),
        "truncation_rule": {
            "study_range": f"degree >= {DEGREE_RANGE[0]}",
            "cap": K_MAX,
            "rule": f"d_used = min(d_observed, {K_MAX})  (right winsorisation, NOT deletion)",
            "tail_alert_threshold": TAIL_ALERT,
        },
        "fanout": fo_tail,
        "fanin": fi_tail,
        "fanout_source": "pooled unique protocol-anchored SOURCE units (v4 + v5, deduped)",
        "fanin_source": "pooled unique protocol-anchored TARGET units (v4 + v5, deduped)",
        "HIGH_TRUNCATION_TAIL": high_tail,
        "HIGH_TRUNCATION_TAIL_note": (
            "P(d > cap) > 0.10 on at least one side; this does not block the experiment "
            "but the manuscript must discuss the truncation bound explicitly."
            if high_tail else
            "P(d > cap) <= 0.10 on both sides; truncation tail is small."),
        "v4_final_report_cross_check": _xref_v4(windows, per_window),
        "v5_manifest_cross_check": _xref_v5(windows, per_window),
    }
    write_json(DIR_DEGREE / "tail_report.json", tail_report)
    log(f"[0A] fanout P(d>8)={fo_tail['p_degree_gt_cap']:.6f} "
        f"fanin P(d>8)={fi_tail['p_degree_gt_cap']:.6f} HIGH_TAIL={high_tail}")

    # ---- degree definition + sampling spec -------------------------------- #
    degree_definition = {
        "stage": "0A",
        "generated_at_utc": utc_now(),
        "fanout": {
            "definition": ("the number of DISTINCT canonical target units associated with "
                           "one protocol-anchored canonical SOURCE unit inside the frozen "
                           "audit window"),
            "dedup_key": ("canonical unit identity as used by the frozen audit: the "
                          "canonical ``src_flow_id`` on the source side and the canonical "
                          "``dst_flow_id`` on the target side, both of which are constructed "
                          "from the canonical ``support_anchors`` transaction hashes"),
            "population": "canonical source units with fan-out >= 2",
        },
        "fanin": {
            "definition": ("the number of DISTINCT canonical source units that route into "
                           "one protocol-anchored canonical TARGET unit"),
            "dedup_key": "same canonical identity rule, read on the target side",
            "population": "canonical target units with fan-in >= 2",
            "empirically_available": True,
            "mirrored_proxy_used": False,
            "note": ("The frozen canonical edge lists are bipartite over canonical source "
                     "AND target ids, so target-side source multiplicity is directly "
                     "recoverable. R7 therefore does NOT fall back to mirroring the fan-out "
                     "distribution, and the generator samples merge degree from the real "
                     "empirical fan-in distribution."),
        },
        "not_assumed_equal": True,
        "cap": K_MAX,
        "range": list(DEGREE_RANGE),
    }
    write_json(DIR_DEGREE / "degree_definition.json", degree_definition)

    def spec(hist: Counter, tail: dict[str, Any], name: str) -> dict[str, Any]:
        tot = int(sum(hist.values()))
        return {
            "name": name,
            "support": [int(d) for d in range(DEGREE_RANGE[0], K_MAX + 1)],
            "pmf": [float(hist[d] / tot) for d in range(DEGREE_RANGE[0], K_MAX + 1)],
            "counts": [int(hist[d]) for d in range(DEGREE_RANGE[0], K_MAX + 1)],
            "n_units": tot,
            "winsorised": True,
            "cap": K_MAX,
            "p_degree_gt_cap": tail["p_degree_gt_cap"],
            "empirical": True,
            "mirrored_proxy_not_empirical_fanin": False,
        }

    sampling_spec = {
        "stage": "0B_input",
        "generated_at_utc": utc_now(),
        "source_windows": ["v4", "v5"],
        "pooling": ("v4/v5 unique protocol-anchored units, deduped on the canonical anchor "
                    "identity; v5 contributes more units and therefore carries more weight "
                    "in the pooled distribution, as required"),
        "split_degree": spec(fo_hist, fo_tail, "split_degree_from_empirical_fanout"),
        "merge_degree": spec(fi_hist, fi_tail, "merge_degree_from_empirical_fanin"),
        "HIGH_TRUNCATION_TAIL": high_tail,
        "hashes": {
            "pooled_degree_hist_truncated.csv": sha256_file(
                DIR_DEGREE / "pooled_degree_hist_truncated.csv"),
            "pooled_fanin_hist_truncated.csv": sha256_file(
                DIR_DEGREE / "pooled_fanin_hist_truncated.csv"),
        },
    }
    write_json(DIR_DEGREE / "degree_sampling_spec.json", sampling_spec)

    summary = {
        "v4": {k: v for k, v in per_window["v4"].items() if not k.startswith("_")},
        "v5": {k: v for k, v in per_window["v5"].items() if not k.startswith("_")},
        "duplicate_units": {k: v for k, v in duplicate.items()
                            if k in ("shared_anchor_count", "duplicate_units_v4",
                                     "duplicate_units_v5", "dedup_applied")},
        "fanout_tail": fo_tail,
        "fanin_tail": fi_tail,
        "HIGH_TRUNCATION_TAIL": high_tail,
    }
    return {"summary": summary, "sampling_spec": sampling_spec, "tail": tail_report,
            "provenance": provenance, "degree_definition": degree_definition}


def _xref_v4(windows: dict[str, Any], per_window: dict[str, Any]) -> dict[str, Any]:
    """Cross-check our recomputed v4 fan-out against the frozen v4 final report."""
    rep_path = V4_ROOT / V4_REPORT_NAME
    if not rep_path.is_file():
        return {"available": False}
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    frozen = {int(k): int(v) for k, v in (rep.get("fanout_degree_distribution") or {}).items()}
    ours = {int(k): int(v) for k, v in per_window["v4"]["fanout_hist_ge2"].items()}
    same = frozen == ours
    return {
        "available": True,
        "frozen_report_fanout_degree_distribution": frozen,
        "r7_recomputed_fanout_degree_distribution": ours,
        "exact_match": same,
        "frozen_n_source_level_fanout_units": rep.get("n_source_level_fanout_units"),
        "r7_recomputed_units": int(sum(ours.values())),
        "note": ("exact agreement confirms the R7 fan-out definition reproduces the frozen "
                 "audit's own definition"
                 if same else
                 "MISMATCH: the R7 recomputation disagrees with the frozen v4 report; the "
                 "frozen report is authoritative and this needs investigation"),
    }


def _xref_v5(windows: dict[str, Any], per_window: dict[str, Any]) -> dict[str, Any]:
    man_path = V5_ROOT / V5_MANIFEST_NAME
    if not man_path.is_file():
        return {"available": False}
    man = json.loads(man_path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "v5_manifest_final_edges": man.get("final_edges"),
        "v5_manifest_final_fanout_units": man.get("final_fanout_units"),
        "r7_recomputed_edges": windows["v5"]["provenance"]["row_count"],
        "r7_recomputed_fanout_units_ge2": per_window["v5"]["fanout_units_ge2"],
        "edges_match": man.get("final_edges") == windows["v5"]["provenance"]["row_count"],
        "fanout_units_match": (man.get("final_fanout_units")
                               == per_window["v5"]["fanout_units_ge2"]),
    }


def main() -> int:
    res = calibrate()
    print(json.dumps(res["summary"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
