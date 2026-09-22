"""Semi-synthetic weak flow labels to stress split / merge / unmatched / noise."""
from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross.config.output_layout import output_file


def _pick_seeds_head(fl: pd.DataFrame, *, rng: random.Random, max_seeds: int = 8) -> list[dict[str, Any]]:
    if fl.empty:
        return []
    oo = fl[fl.get("pattern_type", "").astype(str).str.lower() == "one_to_one"].copy()
    if oo.empty:
        oo = fl.copy()
    oo["_lc"] = pd.to_numeric(oo.get("label_confidence"), errors="coerce").fillna(0.0)
    oo = oo.sort_values("_lc", ascending=False)
    rows = oo.head(max_seeds).to_dict(orient="records")
    rng.shuffle(rows)
    return rows


def _assign_quantile_labels(series: pd.Series, n_bins: int, *, prefix: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if s.nunique() < n_bins:
        ranks = s.rank(method="first")
        nb = max(1, min(n_bins, int(s.nunique())))
        return pd.cut(ranks, bins=nb, labels=[f"{prefix}{i}" for i in range(nb)], include_lowest=True)
    try:
        return pd.qcut(s, q=n_bins, labels=[f"{prefix}{i}" for i in range(n_bins)], duplicates="drop")
    except ValueError:
        ranks = s.rank(method="first")
        return pd.cut(ranks, bins=n_bins, labels=[f"{prefix}{i}" for i in range(n_bins)], include_lowest=True)


def _pick_seeds_stratified(
    fl: pd.DataFrame,
    *,
    rng: random.Random,
    max_seeds: int = 48,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stratified sample: 4 amount quartiles × 3 delay tertiles → 12 cells × 4 seeds = 48."""
    diag: dict[str, Any] = {
        "requested_seeds": int(max_seeds),
        "cells_target_per_cell": 4,
        "cells": [],
        "backfill": [],
    }
    if fl.empty:
        return [], diag

    pool = fl[fl.get("pattern_type", "").astype(str).str.lower() == "one_to_one"].copy()
    if pool.empty:
        pool = fl.copy()
    pool = pool.copy()
    pool["_amt"] = pd.to_numeric(pool.get("src_amount_usd"), errors="coerce").fillna(0.0)
    pool["_delay"] = pd.to_numeric(pool.get("median_delay_sec"), errors="coerce").fillna(0.0)
    pool["_amt_q"] = _assign_quantile_labels(pool["_amt"], 4, prefix="Q")
    pool["_delay_q"] = _assign_quantile_labels(pool["_delay"], 3, prefix="D")
    pool["_cell"] = pool["_amt_q"].astype(str) + "|" + pool["_delay_q"].astype(str)

    n_cells = 12
    per_cell = max(1, int(max_seeds // n_cells))
    diag["cells_target_per_cell"] = per_cell

    chosen_idx: set[int] = set()
    chosen_rows: list[dict[str, Any]] = []

    for cell, grp in pool.groupby("_cell", sort=False):
        idxs = list(grp.index)
        rng.shuffle(idxs)
        take = idxs[:per_cell]
        deficit = per_cell - len(take)
        for i in take:
            chosen_idx.add(int(i))
            chosen_rows.append(pool.loc[i].to_dict())
        diag["cells"].append(
            {
                "cell": str(cell),
                "available": int(len(idxs)),
                "sampled": int(len(take)),
                "deficit": int(max(0, deficit)),
            }
        )

    if len(chosen_rows) < max_seeds:
        remaining = [i for i in pool.index if int(i) not in chosen_idx]
        rng.shuffle(remaining)
        need = max_seeds - len(chosen_rows)
        for i in remaining[:need]:
            chosen_idx.add(int(i))
            chosen_rows.append(pool.loc[i].to_dict())
            diag["backfill"].append({"index": int(i), "cell": str(pool.loc[i, "_cell"])})

    chosen_rows = chosen_rows[:max_seeds]
    rng.shuffle(chosen_rows)
    diag["sampled_total"] = int(len(chosen_rows))
    diag["unique_cells_hit"] = int(len({str(r.get("_cell", "")) for r in chosen_rows}))
    return chosen_rows, diag


def pick_semi_synthetic_seeds(
    fl: pd.DataFrame,
    *,
    rng: random.Random,
    max_seeds: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pick template rows; use stratified grid when ``max_seeds > 8``."""
    if max_seeds > 8:
        return _pick_seeds_stratified(fl, rng=rng, max_seeds=max_seeds)
    rows = _pick_seeds_head(fl, rng=rng, max_seeds=max_seeds)
    return rows, {"mode": "head_confidence", "sampled_total": len(rows)}


# --------------------------------------------------------------------------- #
# R7 family grid
# --------------------------------------------------------------------------- #
# The frozen design stratifies anchors on 4 amount quartiles x 3 delay tertiles
# = 12 cells.  R7 refines the delay axis to sextiles, which splits every frozen cell
# into two halves:
#   * 12 "original" families = the FIRST half of each frozen cell  (delay sextiles 0,2,4)
#   * 12 "new"      families = the SECOND half of each frozen cell (delay sextiles 1,3,5)
# Every family gets its own distinct base anchor, so no base-anchor cluster is shared.
R7_AMOUNT_QUARTILES = 4
R7_DELAY_SEXTILES = 6
R7_INSTANCES_PER_FAMILY = 2


def _r7_family_layout() -> list[dict[str, Any]]:
    """The frozen 24-family layout: (amount quartile, delay sextile) -> family index."""
    out: list[dict[str, Any]] = []
    for a in range(R7_AMOUNT_QUARTILES):          # 0..3
        for d in range(R7_DELAY_SEXTILES):        # 0..5
            out.append({
                "family_index": a * R7_DELAY_SEXTILES + d,
                "amount_quartile": a,
                "delay_sextile": d,
                "delay_tertile": d // 2,
                "half_of_frozen_cell": "first" if d % 2 == 0 else "second",
                "original": (d % 2 == 0),
            })
    return out


def pick_r7_family_seeds(
    fl: pd.DataFrame,
    *,
    rng: random.Random,
    instances_per_family: int = R7_INSTANCES_PER_FAMILY,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pick the R7 template rows: one DISTINCT base anchor per (family, instance).

    Returns rows carrying ``_r7_family`` (0..23) and ``_r7_instance`` (0..k-1).  Anchors
    are drawn without replacement, so the base-anchor clusters of the 24 families are
    disjoint by construction.
    """
    diag: dict[str, Any] = {"mode": "r7_family_grid", "cells": [], "backfill": []}
    if fl.empty:
        return [], diag

    pool = fl[fl.get("pattern_type", "").astype(str).str.lower() == "one_to_one"].copy()
    if pool.empty:
        pool = fl.copy()
    pool["_amt"] = pd.to_numeric(pool.get("src_amount_usd"), errors="coerce").fillna(0.0)
    pool["_delay"] = pd.to_numeric(pool.get("median_delay_sec"), errors="coerce").fillna(0.0)
    pool["_amt_q"] = _assign_quantile_labels(pool["_amt"], R7_AMOUNT_QUARTILES, prefix="Q")
    pool["_delay_q"] = _assign_quantile_labels(pool["_delay"], R7_DELAY_SEXTILES, prefix="S")
    pool["_cell"] = pool["_amt_q"].astype(str) + "|" + pool["_delay_q"].astype(str)

    layout = _r7_family_layout()
    wanted = {f"{('Q' + str(c['amount_quartile']))}|{'S' + str(c['delay_sextile'])}": c
              for c in layout}

    used: set[int] = set()
    chosen: list[tuple[int, dict[str, Any]]] = []
    for cell, meta in sorted(wanted.items(), key=lambda kv: kv[1]["family_index"]):
        grp = pool[pool["_cell"] == cell]
        idxs = [int(i) for i in grp.index]
        rng.shuffle(idxs)
        take = idxs[:instances_per_family]
        diag["cells"].append({
            "cell": cell, "family_index": meta["family_index"],
            "available": len(idxs), "sampled": len(take),
            "deficit": instances_per_family - len(take),
        })
        for inst, i in enumerate(take):
            used.add(i)
            chosen.append((meta["family_index"], {**pool.loc[i].to_dict(),
                                                  "_r7_family": meta["family_index"],
                                                  "_r7_instance": inst,
                                                  "_r7_cell": cell}))

    need = len(layout) * instances_per_family - len(chosen)
    if need > 0:
        remaining = [int(i) for i in pool.index if int(i) not in used]
        rng.shuffle(remaining)
        # backfill deterministically into the families that came up short
        counts: dict[int, int] = {}
        for f, _ in chosen:
            counts[f] = counts.get(f, 0) + 1
        for f in [c["family_index"] for c in layout]:
            while counts.get(f, 0) < instances_per_family and remaining:
                i = remaining.pop()
                counts[f] = counts.get(f, 0) + 1
                chosen.append((f, {**pool.loc[i].to_dict(), "_r7_family": f,
                                   "_r7_instance": counts[f] - 1,
                                   "_r7_cell": str(pool.loc[i, "_cell"])}))
                diag["backfill"].append({"index": i, "family_index": f})

    diag["sampled_total"] = len(chosen)
    diag["family_count"] = len({f for f, _ in chosen})
    diag["instances_per_family"] = instances_per_family
    diag["unique_base_anchors"] = len({str(r.get("src_flow_id")) for _, r in chosen})
    diag["layout"] = layout
    return [r for _, r in chosen], diag


def r7_split_suffix(i: int) -> str:
    """Deterministic leg suffix; index 0/1 reproduce the frozen ``a``/``b`` names."""
    if i < 26:
        return chr(ord("a") + i)
    return f"z{i}"


def r7_template_prefix(base_flow_id: str, family_index: int, instance: int) -> str:
    """R7 instance prefix.

    ``<base_flow_id>__r7fam<FF>__inst<II>``.  ``tpl_of`` of any synthetic id built on
    this prefix returns the prefix itself, so a template id is exactly the instance
    prefix; the family key is the part before ``__inst``.
    """
    return f"{base_flow_id}__r7fam{int(family_index):02d}__inst{int(instance):02d}"


def r7_family_key(template_id: str) -> str:
    s = str(template_id)
    return s.split("__inst")[0] if "__inst" in s else s


def r7_base_anchor(template_id: str) -> str:
    s = str(template_id)
    return s.split("__r7fam")[0] if "__r7fam" in s else s


def build_semi_synthetic_from_flow_labels(
    flow_labels_path: Path,
    flow_label_stats_path: Path,
    out_dir: Path,
    *,
    seed: int = 42,
    max_seeds: int = 8,
    force: bool = False,
    r7_split_degrees: list[int] | None = None,
    r7_merge_degrees: list[int] | None = None,
    r7_family_grid: bool = False,
    r7_instances_per_family: int = R7_INSTANCES_PER_FAMILY,
) -> tuple[Path, Path]:
    """Derive synthetic ``flow_labels``-shaped rows from real labels when data is mostly 1:1.

    When ``predominantly_one_to_one`` is false in stats, generation is skipped unless ``force=True``.
    Writes ``synthetic_seed_stratification.json`` when ``max_seeds > 8``.

    R7 extension (all arguments optional; the defaults reproduce the frozen generator
    byte-for-byte)
    ---------------------------------------------------------------------------------
    ``r7_family_grid``
        24-family grid (4 amount quartiles x 6 delay sextiles), one distinct base anchor
        per (family, instance).
    ``r7_split_degrees`` / ``r7_merge_degrees``
        Per-template split (1 -> d) and merge (d -> 1) degrees, in template order.  When
        omitted the frozen fixed ``1 -> 2`` / ``2 -> 1`` construction is used.  Amount
        allocation generalises the frozen equal-share rule: every split leg receives
        ``amount / d_split`` and every merge source ``amount / d_merge`` (``d = 2`` gives
        exactly the frozen halves).  Everything else -- time perturbation, the +60/+120
        decoys, the unmatched source, the address/evidence/risk construction and all
        bridge-specific mechanisms -- is unchanged.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fl = pd.read_csv(flow_labels_path, dtype=str, keep_default_na=False)
    stats: dict[str, Any] = {}
    if flow_label_stats_path.is_file():
        stats = json.loads(flow_label_stats_path.read_text(encoding="utf-8"))

    predominantly_1_1 = bool(stats.get("predominantly_one_to_one"))
    if not predominantly_1_1 and not force:
        out_csv = output_file(out_dir, "synthetic_flow_labels.csv")
        pd.DataFrame().to_csv(out_csv, index=False)
        out_json = output_file(out_dir, "synthetic_uot_eval_metrics.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "skipped": True,
                    "reason": "not_predominantly_one_to_one",
                    "predominantly_one_to_one": predominantly_1_1,
                    "seed": int(seed),
                    "max_seeds": int(max_seeds),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return out_csv, out_json

    rng = random.Random(int(seed))
    if r7_family_grid:
        seeds, strat_diag = pick_r7_family_seeds(
            fl, rng=rng, instances_per_family=int(r7_instances_per_family))
    else:
        seeds, strat_diag = pick_semi_synthetic_seeds(fl, rng=rng, max_seeds=int(max_seeds))
    strat_diag["random_seed"] = int(seed)
    strat_diag["max_seeds"] = int(max_seeds)
    strat_diag["r7_family_grid"] = bool(r7_family_grid)
    if r7_split_degrees is not None:
        strat_diag["r7_split_degrees"] = [int(x) for x in r7_split_degrees]
    if r7_merge_degrees is not None:
        strat_diag["r7_merge_degrees"] = [int(x) for x in r7_merge_degrees]
    strat_path = output_file(out_dir, "synthetic_seed_stratification.json")
    with open(strat_path, "w", encoding="utf-8") as f:
        json.dump(strat_diag, f, indent=2, ensure_ascii=False)

    rows: list[dict[str, Any]] = []
    truth_pairs: list[list[str]] = []
    truth_unmatched_src: list[str] = []
    noise_pairs: list[list[str]] = []
    clone_records: list[dict[str, Any]] = []
    family_records: list[dict[str, Any]] = []

    def _base(r: dict[str, Any]) -> dict[str, Any]:
        b = deepcopy(r)
        for k in list(b.keys()):
            if str(k).startswith("_"):
                del b[k]
        return b

    for tpl_i, r in enumerate(seeds):
        base = _base(r)
        sf = str(base.get("src_flow_id") or "synth_src")
        df = str(base.get("dst_flow_id") or "synth_dst")
        amt = float(pd.to_numeric(base.get("src_amount_usd"), errors="coerce") or 0.0)

        if r7_family_grid:
            fam_i = int(r.pop("_r7_family", tpl_i))
            inst_i = int(r.pop("_r7_instance", 0))
            cell_str = str(r.get("_r7_cell", ""))
            prefix = r7_template_prefix(sf, fam_i, inst_i)
        else:
            fam_i = tpl_i
            inst_i = 0
            cell_str = ""
            prefix = sf

        d_split = int(r7_split_degrees[tpl_i]) if r7_split_degrees is not None else 2
        d_merge = int(r7_merge_degrees[tpl_i]) if r7_merge_degrees is not None else 2
        share_s = max(amt / d_split, 1e-9)
        share_m = max(amt / d_merge, 1e-9)

        s_split = f"{prefix}__synth_split_src"
        d_list = [f"{df}__synth_split_{r7_split_suffix(k)}" for k in range(d_split)]
        r1 = {
            **base,
            "src_flow_id": s_split,
            "dst_flow_id": d_list[0],
            "pattern_type": "one_to_many",
            "label_source": "semi_synthetic_split",
            "support_tx_pair_count": "1",
            "src_amount_usd": str(amt),
            "dst_amount_usd": str(share_s),
            "matched_src_amount_usd": str(share_s),
            "matched_dst_amount_usd": str(share_s),
            "label_confidence": str(min(0.95, float(pd.to_numeric(base.get("label_confidence"), errors="coerce") or 0.6))),
        }
        for k, d in enumerate(d_list):
            row = dict(r1) if k == 0 else {**r1, "dst_flow_id": d}
            row["dst_flow_id"] = d
            rows.append(row)
            truth_pairs.append([s_split, d])
        clone_records.append(
            {"synthetic_flow_id": s_split, "template_flow_id": sf, "chain": "ETH", "amount_scale": 1.0, "scenario": "split"}
        )
        for d in d_list:
            clone_records.append(
                {"synthetic_flow_id": d, "template_flow_id": df, "chain": "BNB",
                 "amount_scale": 1.0 / d_split, "scenario": "split"})

        s_list = [f"{prefix}__synth_merge_src{k + 1}" for k in range(d_merge)]
        dm = f"{df}__synth_merge_dst"
        m_base = {
            **base,
            "dst_flow_id": dm,
            "pattern_type": "many_to_one",
            "label_source": "semi_synthetic_merge",
            "src_amount_usd": str(share_m),
            "dst_amount_usd": str(amt),
            "matched_src_amount_usd": str(share_m),
            "matched_dst_amount_usd": str(share_m),
        }
        for s in s_list:
            rows.append({**m_base, "src_flow_id": s})
            truth_pairs.append([s, dm])
        for s in s_list:
            clone_records.append(
                {"synthetic_flow_id": s, "template_flow_id": sf, "chain": "ETH",
                 "amount_scale": 1.0 / d_merge, "scenario": "merge"})
        clone_records.append(
            {"synthetic_flow_id": dm, "template_flow_id": df, "chain": "BNB", "amount_scale": 1.0, "scenario": "merge"}
        )

        su = f"{prefix}__synth_unmatched_src"
        hidden = f"{df}__synth_hidden_dst"
        u1 = {
            **base,
            "src_flow_id": su,
            "dst_flow_id": hidden,
            "pattern_type": "one_to_one",
            "label_source": "semi_synthetic_unmatched",
            "dst_amount_usd": "0",
            "matched_dst_amount_usd": "0",
            "flow_mass_ratio_dst": "0",
            "label_confidence": "0.2",
        }
        rows.append(u1)
        truth_unmatched_src.append(su)
        clone_records.extend(
            [
                {"synthetic_flow_id": su, "template_flow_id": sf, "chain": "ETH", "amount_scale": 1.0, "scenario": "unmatched"},
                {"synthetic_flow_id": hidden, "template_flow_id": df, "chain": "BNB", "amount_scale": 0.0, "scenario": "unmatched"},
            ]
        )

        for i in range(2):
            decoy_s = f"{prefix}__synth_noise_src_{i}"
            decoy_d = f"{df}__synth_noise_dst_{i}"
            n1 = {
                **base,
                "src_flow_id": decoy_s,
                "dst_flow_id": decoy_d,
                "pattern_type": "one_to_one",
                "label_source": "semi_synthetic_delay_noise",
                "label_confidence": str(round(0.22 + 0.04 * rng.random(), 4)),
                "median_delay_sec": str(int(pd.to_numeric(base.get("median_delay_sec"), errors="coerce") or 0) + 60 + i * 60),
            }
            rows.append(n1)
            noise_pairs.append([decoy_s, decoy_d])
            clone_records.extend(
                [
                    # Real timestamp perturbation for decoy flows: the destination leg is
                    # shifted (+60s / +120s) so decoy segments do not reuse the template's
                    # original timestamps (root cause C of the retracted run).
                    {"synthetic_flow_id": decoy_s, "template_flow_id": sf, "chain": "ETH",
                     "amount_scale": 1.0, "time_offset_sec": 0.0, "scenario": "delay_noise"},
                    {"synthetic_flow_id": decoy_d, "template_flow_id": df, "chain": "BNB",
                     "amount_scale": 1.0, "time_offset_sec": 60.0 + 60.0 * i, "scenario": "delay_noise"},
                ]
            )

        family_records.append({
            "template_id": prefix,
            "family_index": int(fam_i),
            "family_key": r7_family_key(prefix) if r7_family_grid else sf,
            "instance": int(inst_i),
            "base_anchor_id": sf,
            "base_anchor_dst_id": df,
            "cell": cell_str,
            "split_degree": int(d_split),
            "merge_degree": int(d_merge),
        })

    out_csv = output_file(out_dir, "synthetic_flow_labels.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    metrics: dict[str, Any] = {
        "predominantly_one_to_one": predominantly_1_1,
        "forced": bool(force),
        "seed": int(seed),
        "max_seeds": int(max_seeds),
        "seed_templates_used": int(len(seeds)),
        "synthetic_row_count": int(len(rows)),
        "stratification_path": str(strat_path.resolve()),
        "scenario_counts": {
            "split_edges": int(sum(1 for r in rows if "semi_synthetic_split" in str(r.get("label_source", "")))),
            "merge_edges": int(sum(1 for r in rows if "semi_synthetic_merge" in str(r.get("label_source", "")))),
            "unmatched_labels": int(sum(1 for r in rows if "semi_synthetic_unmatched" in str(r.get("label_source", "")))),
            "noise_edges": int(sum(1 for r in rows if "semi_synthetic_delay_noise" in str(r.get("label_source", "")))),
        },
        "segment_clone_records": clone_records,
        "eval_hints": {
            "truth_flow_pairs": truth_pairs,
            "truth_unmatched_src_flows": truth_unmatched_src,
            "noise_decoy_pairs": noise_pairs,
        },
    }
    if r7_family_grid:
        metrics["r7_family_records"] = family_records
        metrics["r7_split_degrees"] = [int(x) for x in (r7_split_degrees or [])]
        metrics["r7_merge_degrees"] = [int(x) for x in (r7_merge_degrees or [])]
    out_json = output_file(out_dir, "synthetic_uot_eval_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return out_csv, out_json
