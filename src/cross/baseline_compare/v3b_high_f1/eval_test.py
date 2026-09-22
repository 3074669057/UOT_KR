"""Test-set evaluation for v3b."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cross.application.experiments.delay_semantics_utils import _tx_timestamp_lookup
from cross.baseline_compare.v3b_high_f1.masks import eval_mask_ids
from cross.baseline_compare.v3b_high_f1.split import load_split_pairs, truth_dict
from cross.baseline_compare.v3b_high_f1.transport import load_or_compute_transport
from cross.baseline_compare.v3b_high_f1.tunable_decode import (
    OperatingPointParams,
    build_candidate_cache,
    decode_with_params,
    evaluate_on_truth,
    frozen_v3_params,
    predictions_df,
)
from cross.shared.normalize import norm_addr
from cross.shared.transfers import eth_df_to_src_txs


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_selected(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "dev_tune" / "selected_operating_points.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run --tune-dev first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def eval_test(*, output_dir: Path, eth_csv: Path, bnb_csv: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    selected = _load_selected(output_dir)
    test_df = load_split_pairs(output_dir, "test")
    test_truth = truth_dict(test_df)
    frozen = frozen_v3_params()
    global_params = OperatingPointParams(**selected["global_best"]["selected_params"])

    eth_df = pd.read_csv(eth_csv, dtype=str, low_memory=False)
    bnb_df = pd.read_csv(bnb_csv, dtype=str, low_memory=False)
    eth_ts, bnb_ts, _ = _tx_timestamp_lookup(eth_df, bnb_df)
    dst_norm = bnb_df.copy()
    if "hash" in dst_norm.columns:
        dst_norm["hash"] = dst_norm["hash"].astype(str).map(norm_addr)

    test_dir = output_dir / "test_eval"
    results: dict[str, Any] = {}

    for mask_id in eval_mask_ids():
        mask_dir = test_dir / mask_id
        required = (
            mask_dir / "frozen_eval.json",
            mask_dir / "global_best_eval.json",
            mask_dir / "per_mask_best_eval.json",
        )
        if all(p.is_file() for p in required):
            print(f"v3b test eval: {mask_id} (skip, complete)", flush=True)
            frozen_ev = json.loads((mask_dir / "frozen_eval.json").read_text(encoding="utf-8"))
            global_ev = json.loads((mask_dir / "global_best_eval.json").read_text(encoding="utf-8"))
            per_ev = json.loads((mask_dir / "per_mask_best_eval.json").read_text(encoding="utf-8"))
            results[mask_id] = {
                "frozen_f1": frozen_ev.get("pair_f1"),
                "global_best_f1": global_ev.get("pair_f1"),
                "per_mask_best_f1": per_ev.get("pair_f1"),
            }
            continue

        print(f"v3b test eval: {mask_id} ...", flush=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        per_params_raw = selected["per_mask_best"].get(mask_id, {}).get("selected_params")
        per_params = OperatingPointParams(**per_params_raw) if per_params_raw else global_params

        p, eth_f, bnb_f, _, _ = load_or_compute_transport(mask_id=mask_id, output_dir=output_dir)
        flow_txs = {norm_addr(str(h)) for f in eth_f for h in (f.get("tx_hashes") or [])}
        src_all = eth_df_to_src_txs(eth_df)
        if flow_txs:
            src_all = src_all[src_all["txhash"].astype(str).map(norm_addr).isin(flow_txs)].reset_index(drop=True)

        caches = build_candidate_cache(
            truth=test_truth,
            p=p,
            eth_flows=eth_f,
            bnb_flows=bnb_f,
            src_all=src_all,
            dst_norm=dst_norm,
            eth_ts=eth_ts,
            bnb_ts=bnb_ts,
        )

        for op_type, params in (
            ("frozen", frozen),
            ("global_best", global_params),
            ("per_mask_best", per_params),
        ):
            ev = evaluate_on_truth(
                caches=caches,
                truth=test_truth,
                params=params,
                eth_flows=eth_f,
                bnb_flows=bnb_f,
                eth_ts=eth_ts,
                bnb_ts=bnb_ts,
                label_df=test_df,
                split="test",
                mask_id=mask_id,
                operating_point_type=op_type,
            )
            _write_json(mask_dir / f"{op_type}_eval.json", ev)

            mapping = {}
            for c in caches:
                dst, _ = decode_with_params(
                    c, params, eth_flows=eth_f, bnb_flows=bnb_f, eth_ts=eth_ts, bnb_ts=bnb_ts
                )
                mapping[c.src_tx] = dst
            predictions_df(mapping, test_truth).to_csv(
                mask_dir / f"predictions_{op_type}.csv", index=False
            )

        results[mask_id] = {
            "frozen_f1": json.loads((mask_dir / "frozen_eval.json").read_text())["pair_f1"],
            "global_best_f1": json.loads((mask_dir / "global_best_eval.json").read_text())["pair_f1"],
            "per_mask_best_f1": json.loads((mask_dir / "per_mask_best_eval.json").read_text())["pair_f1"],
        }

    report = {"generated_at_utc": _utc(), "n_masks": len(results), "results": results}
    _write_json(test_dir / "test_eval_summary.json", report)
    return report
