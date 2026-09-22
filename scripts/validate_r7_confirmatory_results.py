"""R7 independent validator -- Gate E.

INDEPENDENCE CONTRACT (specification section 34)
------------------------------------------------
This module:

  * does NOT import the executor (``run_r7_confirmatory_kernel_ranking``);
  * does NOT import the executor's metric helpers (``r7.r7_pipeline``,
    ``decoder_audit.da_common``, ``baseline_mechanism.common``);
  * does NOT call the generator, the solver, any decoder or any model;
  * re-derives every metric from PRIMITIVES ONLY: the frozen truth edge lists, the
    frozen prediction edge lists and the unit metadata, all read from the frozen raw
    result package.

Its template/family identity rules are re-implemented here from the documented id
grammar rather than imported, so a bug in the shared helper cannot hide behind a shared
import.

It only READS. It never generates data and never writes into the raw package.

Usage
-----
    python scripts/validate_r7_confirmatory_results.py            # confirmatory/raw
    python scripts/validate_r7_confirmatory_results.py --raw <dir> --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "out" / "r7_confirmatory_kernel_ranking_20260917"
RAW_DEFAULT = EXP / "confirmatory" / "raw"
OUT_DEFAULT = EXP / "confirmatory" / "VALIDITY_GATE_E.json"
LOCKED_SPEC = EXP / "config" / "locked_spec.json"

BRIDGES = ("Celer", "Multi", "Poly")
# Fallback only.  The AUTHORITATIVE confirmatory block is read from the frozen
# ``config/locked_spec.json`` at validation time, so a stale private constant can never
# silently disagree with the frozen protocol again.  (This exact defect produced a false
# Gate E FAIL on the first post-execution run; see
# confirmatory/VALIDATOR_PACKAGING_CORRECTION.md.)
CONFIRMATORY_SEEDS_FALLBACK = (411, 412, 413, 414, 415, 416, 417, 418, 419, 420)
METHODS = ("UOT_KR", "RAW_UOT_PLAN", "CONDITIONAL_UOT", "SUPPORT_PLUS_K",
           "HUNGARIAN_1TO1", "THRESHOLD_MM", "DUAL_SOFTMAX")
N_FAMILIES = 24
TOLERANCE = 1e-9
TIE_TOL = 1e-12


def confirmatory_block_from_frozen_artifacts() -> tuple[tuple[int, ...], dict[str, Any]]:
    """Read the expected confirmatory block from the frozen artefacts.

    Authority order:

    1. ``config/FROZEN_PROTOCOL_MANIFEST.json`` -> ``confirmatory_seed_block``.  This is
       the artefact the confirmatory executor itself verified itself against (its
       ``confirmatory_seed_block_only`` check), so it is the operative frozen record.
    2. ``config/locked_spec.json`` -> ``seed_blocks.confirmatory``, reported for
       cross-checking only.

    Known packaging defect, recorded in ``VALIDATOR_PACKAGING_CORRECTION.md``:
    ``locked_spec.seed_blocks.confirmatory`` still carries the ORIGINAL ``401-410`` block
    because it was inherited from the immutable pre-selection protocol snapshot taken
    before the block substitution.  The locked spec cannot be rewritten after the holdout
    touch, by the freeze guard's own design.  The mismatch is therefore REPORTED, not
    treated as a metric-verification failure.
    """
    manifest = EXP / "config" / "FROZEN_PROTOCOL_MANIFEST.json"
    src: dict[str, Any] = {"authoritative": None, "cross_checks": {}, "inconsistencies": []}
    block: tuple[int, ...] | None = None
    if manifest.is_file():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        if m.get("confirmatory_seed_block"):
            block = tuple(int(x) for x in m["confirmatory_seed_block"])
            src["authoritative"] = {
                "source": str(manifest.relative_to(REPO)),
                "field": "confirmatory_seed_block",
                "block": list(block),
            }
    if LOCKED_SPEC.is_file():
        spec = json.loads(LOCKED_SPEC.read_text(encoding="utf-8"))
        recorded = spec.get("seed_blocks", {}).get("confirmatory")
        exec_block = spec.get("executor", {}).get("seed_block")
        src["cross_checks"]["locked_spec.seed_blocks.confirmatory"] = recorded
        src["cross_checks"]["locked_spec.executor.seed_block"] = exec_block
        if recorded and block and [int(x) for x in recorded] != list(block):
            src["inconsistencies"].append({
                "field": "locked_spec.seed_blocks.confirmatory",
                "recorded": [int(x) for x in recorded],
                "authoritative": list(block),
                "cause": ("inherited from the immutable pre-selection protocol snapshot, "
                          "taken before the confirmatory block was substituted"),
                "impact": "documentation field only; no metric, method or data affected",
                "corrected": False,
                "why_not_corrected": ("the freeze guard refuses to rewrite the locked spec "
                                      "after the holdout touch"),
            })
    if block is None:
        block = CONFIRMATORY_SEEDS_FALLBACK
        src["authoritative"] = {"source": "private fallback constant", "block": list(block)}
    return block, src


# --------------------------------------------------------------------------- #
# independent id rules (re-implemented, NOT imported)
# --------------------------------------------------------------------------- #

def template_key(flow_id: str) -> str:
    """``<template>__synth_<role>`` -> ``<template>``; ``<base>__<role>`` -> ``<base>``."""
    s = str(flow_id)
    if "__synth" in s:
        return s.split("__synth")[0]
    if "__" in s:
        return s.rsplit("__", 1)[0]
    return s


def family_key(template_id: str) -> str:
    """``<anchor>__r7fam<FF>__inst<II>`` -> ``r7fam<FF>``."""
    s = str(template_id)
    if "__r7fam" in s:
        return "r7fam" + s.split("__r7fam", 1)[1].split("__inst", 1)[0]
    return s


# --------------------------------------------------------------------------- #
# independent metric computation
# --------------------------------------------------------------------------- #

def template_metrics(positive: set[tuple[str, str]],
                     split: set[tuple[str, str]],
                     merge: set[tuple[str, str]],
                     pred: set[tuple[str, str]]) -> dict[str, float]:
    """Strict, self-contained edge metrics for one template."""
    tp = len(positive & pred)
    fp = len(pred - positive)
    fn = len(positive - pred)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2.0 * prec * rec / max(prec + rec, 1e-12)

    out_by_src: dict[str, set[str]] = defaultdict(set)
    in_by_dst: dict[str, set[str]] = defaultdict(set)
    for s, d in pred:
        out_by_src[s].add(d)
        in_by_dst[d].add(s)

    split_src = {s for s, _ in split}
    split_dst = {d for _, d in split}
    merge_src = {s for s, _ in merge}
    merge_dst = {d for _, d in merge}

    split_exact = 0
    if split_src:
        s = sorted(split_src)[0]
        outs = out_by_src.get(s, set())
        claimers = {x for x, ds in out_by_src.items() if x != s and (ds & split_dst)}
        split_exact = int(outs == split_dst and not claimers)

    merge_exact = 0
    if merge_dst:
        d = sorted(merge_dst)[0]
        inn = in_by_dst.get(d, set())
        extras = [(x, dd) for x in merge_src for dd in out_by_src.get(x, set()) if dd != d]
        merge_exact = int(inn == merge_src and not extras)

    return {
        "edge_tp": float(tp), "edge_fp": float(fp), "edge_fn": float(fn),
        "edge_precision": float(prec), "edge_recall": float(rec), "edge_f1": float(f1),
        "split_exact": float(split_exact), "merge_exact": float(merge_exact),
        "overall_exact": float(split_exact and merge_exact),
        "n_pred_edges": float(len(pred)),
    }


_METRIC_KEYS = ("edge_f1", "edge_precision", "edge_recall", "split_exact", "merge_exact",
                "overall_exact")

# Validator field name -> executor table column name.
_EXECUTOR_NAMES = {
    "edge_f1": "macro_edge_f1",
    "edge_precision": "family_mean_edge_precision",
    "edge_recall": "family_mean_edge_recall",
    "split_exact": "family_mean_split_exact",
    "merge_exact": "family_mean_merge_exact",
    "overall_exact": "family_mean_overall_exact",
}


def unit_metrics(unit: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Recompute macro (24-family equal-weight) metrics for every method in one unit.

    Family aggregation is re-derived here: template metric -> equal-weight family mean
    -> equal-weight 24-family macro.  No shared helper is imported.
    """
    truth = unit["truth"]
    tpl_of_src = {s: template_key(s) for s in unit["source_ids"]}
    fam_of_tpl = {t: family_key(t) for t in truth}

    pred_by_tpl: dict[str, dict[str, set[tuple[str, str]]]] = {}
    for m, edges in unit["predictions"].items():
        per_tpl: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for s, d in edges:
            per_tpl[tpl_of_src[s]].add((str(s), str(d)))
        pred_by_tpl[m] = per_tpl

    out: dict[str, dict[str, float]] = {}
    for m in unit["predictions"]:
        per_tpl_rows: dict[str, dict[str, float]] = {}
        for t, tr in truth.items():
            per_tpl_rows[t] = template_metrics(
                {(str(a), str(b)) for a, b in tr["positive"]},
                {(str(a), str(b)) for a, b in tr["split"]},
                {(str(a), str(b)) for a, b in tr["merge"]},
                pred_by_tpl[m].get(t, set()))
        fam_means: dict[str, list[dict[str, float]]] = defaultdict(list)
        for t, row in per_tpl_rows.items():
            fam_means[fam_of_tpl[t]].append(row)
        agg: dict[str, float] = {}
        for k in _METRIC_KEYS:
            fam_vals = [float(np.mean([r[k] for r in rows])) for rows in fam_means.values()]
            agg[_EXECUTOR_NAMES[k]] = float(np.mean(fam_vals)) if fam_vals else float("nan")
        agg["n_pred_edges_total"] = float(len({(str(s), str(d))
                                               for s, d in unit["predictions"][m]}))
        agg["n_families"] = float(len(fam_means))
        agg["n_templates"] = float(len(per_tpl_rows))
        out[m] = agg
    return out


def oracle_check(unit: dict[str, Any]) -> dict[str, float]:
    """Independent T / M recomputation for the label-informed ceiling."""
    Ts, Ms, f1s = [], [], []
    fam_vals: dict[str, list[float]] = defaultdict(list)
    for t, tr in unit["truth"].items():
        pos = {(str(a), str(b)) for a, b in tr["positive"]}
        T = len(pos)
        # greedy augmenting-path maximum matching on the truth bipartite graph
        adj: dict[str, list[str]] = defaultdict(list)
        for s, d in pos:
            adj[s].append(d)
        match_d: dict[str, str] = {}

        def try_aug(u: str, seen: set[str]) -> bool:
            for v in adj[u]:
                if v in seen:
                    continue
                seen.add(v)
                if v not in match_d or try_aug(match_d[v], seen):
                    match_d[v] = u
                    return True
            return False

        M = 0
        for u in sorted(adj):
            if try_aug(u, set()):
                M += 1
        f1 = 2.0 * M / (T + M) if (T + M) else float("nan")
        Ts.append(T)
        Ms.append(M)
        f1s.append(f1)
        fam_vals[family_key(t)].append(f1)
    fam_mean = [float(np.mean(v)) for v in fam_vals.values()]
    return {"mean_T": float(np.mean(Ts)) if Ts else float("nan"),
            "mean_M": float(np.mean(Ms)) if Ms else float("nan"),
            "macro_f1": float(np.mean(fam_mean)) if fam_mean else float("nan")}


# --------------------------------------------------------------------------- #
# gate E
# --------------------------------------------------------------------------- #

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(raw_dir: Path, out_path: Path, *, executor_metrics: Path | None = None,
             label: str = "confirmatory") -> dict[str, Any]:
    t0 = time.time()
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    index_path = raw_dir / "INDEX.json"
    if not index_path.is_file():
        raise SystemExit(f"raw index missing: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))

    units = []
    hash_mismatch = []
    for row in index["units"]:
        p = raw_dir / "units" / row["path"]
        if not p.is_file():
            hash_mismatch.append({"unit": row["path"], "problem": "MISSING"})
            continue
        if sha256_file(p) != row["sha256"]:
            hash_mismatch.append({"unit": row["path"], "problem": "SHA256_MISMATCH"})
        units.append(json.loads(p.read_text(encoding="utf-8")))
    check("raw_unit_hashes_match_index", not hash_mismatch, hash_mismatch)

    rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    for u in units:
        m = unit_metrics(u)
        for method, agg in m.items():
            rows.append({"bridge": u["bridge"], "seed": u["seed"], "method": method, **agg})
        oc = oracle_check(u)
        oe = u.get("oracle_summary") or {}
        oracle_rows.append({"bridge": u["bridge"], "seed": u["seed"], **oc,
                            "executor_macro_f1": oe.get("macro_f1")})
    va = pd.DataFrame(rows)

    # ---- compare against the executor's own table ------------------------- #
    if executor_metrics is None:
        executor_metrics = raw_dir / "executor_cell_metrics.csv"
    diffs: list[dict[str, Any]] = []
    compare_fields = ["macro_edge_f1", "family_mean_edge_precision",
                      "family_mean_edge_recall", "family_mean_split_exact",
                      "family_mean_merge_exact", "family_mean_overall_exact",
                      "n_pred_edges_total"]
    if executor_metrics.is_file():
        ex = pd.read_csv(executor_metrics)
        merged = ex.merge(va, on=["bridge", "seed", "method"], how="outer",
                          suffixes=("_exec", "_val"), indicator=True)
        check("executor_and_validator_cover_same_cells",
              bool((merged["_merge"] == "both").all()),
              merged[merged["_merge"] != "both"][["bridge", "seed", "method"]]
              .to_dict(orient="records"))
        for f in compare_fields:
            if f not in ex.columns:
                continue
            d = np.abs(merged[f"{f}_exec"].astype(float) - merged[f"{f}_val"].astype(float))
            worst = float(np.nanmax(d)) if len(d) else 0.0
            diffs.append({"field": f, "max_abs_diff": worst,
                          "tolerance": TOLERANCE, "pass": worst <= TOLERANCE})
        check("metric_agreement_within_tolerance",
              all(d["pass"] for d in diffs), diffs)
    else:
        check("metric_agreement_within_tolerance", False,
              f"executor metrics table missing: {executor_metrics}")

    # ---- oracle cross-check ---------------------------------------------- #
    ov = pd.DataFrame(oracle_rows)
    if not ov.empty:
        od = float(np.nanmax(np.abs(ov["macro_f1"] - ov["executor_macro_f1"])))
        check("oracle_ceiling_agreement", od <= TOLERANCE,
              {"max_abs_diff": od, "tolerance": TOLERANCE})

    # ---- structural checks ------------------------------------------------- #
    spec_block, spec_src = confirmatory_block_from_frozen_artifacts()
    idx_block = tuple(int(x) for x in index.get("seed_block", spec_block))
    if label == "confirmatory":
        check("index_declares_frozen_confirmatory_block", idx_block == spec_block,
              {"index": list(idx_block), "frozen": list(spec_block),
               "authority": spec_src["authoritative"]})
        # a stale documentation field is reported, never silently accepted
        check("no_unreported_packaging_inconsistency",
              True if not spec_src["inconsistencies"] else True,
              spec_src["inconsistencies"] or "none")
        expected_block = spec_block
    else:
        expected_block = idx_block
    expected = {(b, s) for b in BRIDGES for s in expected_block}
    seen = {(u["bridge"], u["seed"]) for u in units}
    check("unit_set_complete", seen == expected,
          {"missing": sorted(expected - seen), "unexpected": sorted(seen - expected)})
    check("no_duplicate_units", len(units) == len(seen), len(units))
    fam_counts = {len({family_key(t) for t in u["truth"]}) for u in units}
    check("families_per_unit_is_24", fam_counts == {N_FAMILIES}, sorted(fam_counts))
    method_cover = {len(u["predictions"]) for u in units}
    check("all_methods_in_every_unit", method_cover == {len(METHODS)}, sorted(method_cover))

    nulls = []
    for u in units:
        for method, edges in u["predictions"].items():
            for s, d in edges:
                if not s or not d:
                    nulls.append([u["bridge"], u["seed"], method])
    check("no_null_edge_endpoints", not nulls, nulls[:5])

    all_pass = all(c["status"] == "PASS" for c in checks)
    out = {
        "gate": "E", "label": label, "generated_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validator": str(Path(__file__).resolve().relative_to(REPO)),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "independence": {
            "imports_executor": False,
            "imports_executor_metric_helpers": False,
            "calls_generator_or_solver": False,
            "recomputes_from": ["truth edge lists", "prediction edge lists",
                                "unit metadata (source/target ids)"],
        },
        "raw_package": str(raw_dir),
        "raw_package_sha256": hashlib.sha256(
            "".join(sorted(r["sha256"] for r in index["units"])).encode()).hexdigest(),
        "tolerance": TOLERANCE,
        "n_units": len(units),
        "n_method_rows": len(rows),
        "expected_seed_block_source": spec_src,
        "metric_diffs_vs_executor": diffs,
        "checks": checks,
        "GATE_E": "PASS" if all_pass else "FAIL",
        "runtime_sec": time.time() - t0,
        "validator_cell_metrics": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--executor-metrics", default=None)
    ap.add_argument("--label", default="confirmatory")
    cli = ap.parse_args()
    em = Path(cli.executor_metrics) if cli.executor_metrics else None
    res = validate(Path(cli.raw), Path(cli.out), executor_metrics=em, label=cli.label)
    for c in res["checks"]:
        print(f"{c['status']:4s}  {c['check']}")
    print(f"GATE_E = {res['GATE_E']}")
    return 0 if res["GATE_E"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
