"""Dev-set operating point search for v3b."""
from __future__ import annotations

import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cross.application.experiments.delay_semantics_utils import _tx_timestamp_lookup
from cross.baseline_compare.v3b_high_f1.constants import (
    ABSTENTION_POLICY_GRID,
    CONFIDENCE_THRESHOLD_GRID,
    COVERAGE_THRESHOLD_GRID,
    MARGIN_THRESHOLD_GRID,
    TIME_ADMISSIBLE_WINDOW_SEC_GRID,
    TOP_K_GRID,
    TRANSPORT_MASS_THRESHOLD_GRID,
    TX_CVR_MAX,
)
from cross.baseline_compare.v3b_high_f1.masks import eval_mask_ids
from cross.baseline_compare.v3b_high_f1.split import load_split_pairs, truth_dict
from cross.baseline_compare.v3b_high_f1.transport import load_or_compute_transport
from cross.baseline_compare.v3b_high_f1.tunable_decode import (
    OperatingPointParams,
    build_candidate_cache,
    evaluate_on_truth,
    frozen_v3_params,
)
from cross.shared.normalize import norm_addr
from cross.shared.transfers import eth_df_to_src_txs


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def param_grid() -> tuple[list[OperatingPointParams], str]:
    full = os.environ.get("V3B_FULL_GRID", "0") == "1"
    if full:
        top_k = TOP_K_GRID
        tm = TRANSPORT_MASS_THRESHOLD_GRID
        conf = CONFIDENCE_THRESHOLD_GRID
        marg = MARGIN_THRESHOLD_GRID
        cov = COVERAGE_THRESHOLD_GRID
        tw = TIME_ADMISSIBLE_WINDOW_SEC_GRID
        mode = "full_spec_grid"
    else:
        top_k = [1, 3, 5, 10]
        tm = [0.0, 1e-3]
        conf = [0.0, 0.3, 0.5]
        marg = [0.0, 0.1]
        cov = [0.0]
        tw = [7200, 43200]
        mode = "pragmatic_grid_default"

    grid: list[OperatingPointParams] = []
    for top_k_v, tm_v, conf_v, marg_v, cov_v, tw_v, pol in itertools.product(
        top_k, tm, conf, marg, cov, tw, ABSTENTION_POLICY_GRID
    ):
        grid.append(
            OperatingPointParams(
                top_k=int(top_k_v),
                transport_mass_threshold=float(tm_v),
                confidence_threshold=float(conf_v),
                margin_threshold=float(marg_v),
                coverage_threshold=float(cov_v),
                time_admissible_window_sec=float(tw_v),
                abstention_policy=str(pol),
            )
        )
    return grid, mode


def _score_eval(ev: dict[str, Any]) -> tuple:
    f1 = float(ev.get("pair_f1") or 0.0)
    tx_cvr = float(ev.get("tx_CVR") or 0.0)
    prec = float(ev.get("pair_precision") or 0.0)
    cov = float(ev.get("coverage") or 0.0)
    params = ev.get("selected_params") or {}
    simplicity = OperatingPointParams(**params).simplicity_score() if params else 99
    return (f1, -tx_cvr, prec, cov, -simplicity)


def _pick_best(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [
        c
        for c in candidates
        if c.get("n_predicted", 0) > 0
        and float(c.get("tx_CVR") or 1.0) <= TX_CVR_MAX
    ]
    if not valid:
        return None
    return max(valid, key=_score_eval)


def tune_dev(*, output_dir: Path, eth_csv: Path, bnb_csv: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    dev_df = load_split_pairs(output_dir, "dev")
    dev_truth = truth_dict(dev_df)
    grid, grid_mode = param_grid()
    frozen = frozen_v3_params()

    eth_df = pd.read_csv(eth_csv, dtype=str, low_memory=False)
    bnb_df = pd.read_csv(bnb_csv, dtype=str, low_memory=False)
    eth_ts, bnb_ts, _ = _tx_timestamp_lookup(eth_df, bnb_df)
    dst_norm = bnb_df.copy()
    if "hash" in dst_norm.columns:
        dst_norm["hash"] = dst_norm["hash"].astype(str).map(norm_addr)

    per_mask_best: dict[str, Any] = {}
    global_scores: dict[str, list[float]] = {}

    tune_dir = output_dir / "dev_tune"
    tune_dir.mkdir(parents=True, exist_ok=True)

    for mask_id in eval_mask_ids():
        print(f"v3b dev tune: {mask_id} ...", flush=True)
        p, eth_f, bnb_f, _, _ = load_or_compute_transport(mask_id=mask_id, output_dir=output_dir)
        flow_txs = {norm_addr(str(h)) for f in eth_f for h in (f.get("tx_hashes") or [])}
        src_all = eth_df_to_src_txs(eth_df)
        if flow_txs:
            src_all = src_all[src_all["txhash"].astype(str).map(norm_addr).isin(flow_txs)].reset_index(drop=True)

        full_truth = dev_truth
        caches = build_candidate_cache(
            truth=full_truth,
            p=p,
            eth_flows=eth_f,
            bnb_flows=bnb_f,
            src_all=src_all,
            dst_norm=dst_norm,
            eth_ts=eth_ts,
            bnb_ts=bnb_ts,
            max_k=max(TOP_K_GRID),
        )
        dev_caches = [c for c in caches if c.src_tx in dev_truth]

        results: list[dict[str, Any]] = []
        for params in grid:
            ev = evaluate_on_truth(
                caches=dev_caches,
                truth=dev_truth,
                params=params,
                eth_flows=eth_f,
                bnb_flows=bnb_f,
                eth_ts=eth_ts,
                bnb_ts=bnb_ts,
                label_df=dev_df,
                split="dev",
                mask_id=mask_id,
                operating_point_type="dev_search",
            )
            results.append(ev)
            key = json.dumps(params.to_dict(), sort_keys=True)
            global_scores.setdefault(key, []).append(float(ev.get("pair_f1") or 0.0))

        best = _pick_best(results)
        frozen_ev = evaluate_on_truth(
            caches=dev_caches,
            truth=dev_truth,
            params=frozen,
            eth_flows=eth_f,
            bnb_flows=bnb_f,
            eth_ts=eth_ts,
            bnb_ts=bnb_ts,
            label_df=dev_df,
            split="dev",
            mask_id=mask_id,
            operating_point_type="frozen_reference",
        )
        per_mask_best[mask_id] = {
            "best_dev_eval": best,
            "frozen_dev_eval": frozen_ev,
            "n_grid_evaluated": len(grid),
            "n_valid_under_tx_cvr": sum(1 for r in results if r.get("tx_cvr_constraint_satisfied")),
        }
        _write_json(tune_dir / f"{mask_id}_dev_search.json", per_mask_best[mask_id])
        print(
            f"  done {mask_id}: best_dev_f1={(best or {}).get('pair_f1')} frozen={(frozen_ev.get('pair_f1'))}",
            flush=True,
        )

    global_best_key = max(global_scores, key=lambda k: sum(global_scores[k]) / len(global_scores[k]))
    global_best_params = json.loads(global_best_key)
    global_best = OperatingPointParams(**global_best_params)
    mean_global_f1 = sum(global_scores[global_best_key]) / len(global_scores[global_best_key])

    out = {
        "generated_at_utc": _utc(),
        "grid_mode": grid_mode,
        "n_masks": len(eval_mask_ids()),
        "grid_size": len(grid),
        "tx_cvr_max": TX_CVR_MAX,
        "per_mask_best": {
            mid: {
                "selected_params": (per_mask_best[mid]["best_dev_eval"] or {}).get("selected_params"),
                "dev_pair_f1": (per_mask_best[mid]["best_dev_eval"] or {}).get("pair_f1"),
                "dev_tx_CVR": (per_mask_best[mid]["best_dev_eval"] or {}).get("tx_CVR"),
                "frozen_dev_f1": per_mask_best[mid]["frozen_dev_eval"].get("pair_f1"),
            }
            for mid in per_mask_best
        },
        "global_best": {
            "selected_params": global_best.to_dict(),
            "mean_dev_f1_across_masks": mean_global_f1,
        },
    }
    _write_json(output_dir / "dev_tune" / "selected_operating_points.json", out)
    return out
