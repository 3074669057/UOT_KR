"""Three-bridge symmetric masking ladder (raw tx-pair full-set, held-out).

For each bridge, the dev-selected configuration of each method is FROZEN from the
unmasked run (``*_selection.json``) and reused unchanged at test time.  Masking
only removes features from the test inputs; nothing is re-tuned per mask.

Mask levels:
  full                   no change
  no_receiver            source_receiver and candidate receiver cleared
  no_amount              source_amount_raw and candidate amount_raw zeroed
  no_receiver_no_amount  receiver cleared + amount zeroed
  no_token               source_token_address and candidate token_address cleared
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from run_rc_uot_q_multi_bridge import (  # noqa: E402
    load_bridge_cached,
    normalize_baseline_amounts,
    predict_abctracer_style,
    prepare_baseline_candidates,
    prepare_baseline_source,
    prepare_rc,
    truth_for,
    OLD,
)
from cross.application.experiments import ec_uot_q_final  # noqa: E402
from cross.application.experiments.ec_uot_q_final import evaluate_full_set, predict  # noqa: E402

OUT = REPO / "out" / "multi_bridge_expansion" / "masking_ladder"
BRIDGES = ("Celer", "Multi", "Poly")
METHODS = ("RC-UOT-Q", "Connector-style", "ABCTracer-style")
MASKS = ("full", "no_receiver", "no_amount", "no_receiver_no_amount")


def status_of(m: dict[str, Any]) -> str:
    if m.get("n_predicted", 0) == 0:
        return "NO_PREDICTIONS"
    return "OK"


def run_method(bridge: str, cached: dict[str, Any], method: str, mask: str, frozen: dict[str, Any], token_map: dict[str, str]) -> dict[str, Any]:
    split = cached["split"]
    source = cached["source"].copy()
    receiver = dict(cached["receiver"])
    test_candidates = cached["test_candidates"].copy()

    if mask in ("no_receiver", "no_receiver_no_amount"):
        receiver = {k: "" for k in receiver}
        source["source_receiver"] = ""
    if mask in ("no_amount", "no_receiver_no_amount"):
        source["source_amount_raw"] = 0.0
        test_candidates["amount_raw"] = 0.0
    if mask == "no_token":
        source["source_token_address"] = ""
        test_candidates["token_address"] = ""

    test_source = source.loc[source["split"] == "test"].copy()
    test_truth = truth_for(split, "test")

    if method == "RC-UOT-Q":
        src, cand = prepare_rc(test_source, test_candidates)
        ec_uot_q_final._PREDICT_CACHE.clear()
        pred, _ = predict(src, cand, frozen)
        metrics = evaluate_full_set(test_truth, pred)
    elif method == "Connector-style":
        src = prepare_baseline_source(test_source, receiver)
        cand = prepare_baseline_candidates(test_candidates)
        src, cand = normalize_baseline_amounts(src, cand)
        cfg = OLD.base_config(float(frozen.get("confidence_threshold", 0.0)))
        cfg["window_h"] = float(frozen.get("window_h", 6.0))
        pred, conf = OLD.predict_exact(src, cand, token_map, cfg)
        thr = float(frozen.get("confidence_threshold", 0.0))
        pred = {s: (d if d is not None and conf.get(s, 0.0) >= thr else None) for s, d in pred.items()}
        metrics = evaluate_full_set(test_truth, pred)
    elif method == "ABCTracer-style":
        src = prepare_baseline_source(test_source, receiver)
        cand = prepare_baseline_candidates(test_candidates)
        src, cand = normalize_baseline_amounts(src, cand)
        cfg = {"window_h": float(frozen.get("window_h", 6.0)), "before_sec": float(frozen.get("before_sec", 900.0))}
        pred, _ = predict_abctracer_style(src, cand, cfg)
        metrics = evaluate_full_set(test_truth, pred)
    else:
        raise ValueError(method)

    metrics["status"] = status_of(metrics)
    return metrics


def load_frozen_configs(bridge: str) -> dict[str, dict[str, Any]]:
    root = REPO / "out" / "multi_bridge_expansion" / bridge
    out: dict[str, dict[str, Any]] = {}
    for method, key in [("RC-UOT-Q", "rc_uot_q"), ("Connector-style", "connector"), ("ABCTracer-style", "abctracer")]:
        d = json.loads((root / f"{key}_selection.json").read_text(encoding="utf-8"))
        out[method] = d["selected_config"]
    return out


def build_token_map(bridge: str, cached: dict[str, Any]) -> dict[str, str]:
    split = cached["split"]
    all_c = pd.concat([cached["dev_candidates"], cached["test_candidates"]], ignore_index=True).drop_duplicates("candidate_tx_hash")
    return OLD.infer_token_map(split, all_c)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for bridge in BRIDGES:
        cached = load_bridge_cached(bridge)
        frozen = load_frozen_configs(bridge)
        token_map = build_token_map(bridge, cached)
        for mask in MASKS:
            for method in METHODS:
                m = run_method(bridge, cached, method, mask, frozen[method], token_map)
                rows.append({
                    "bridge": bridge,
                    "mask": mask,
                    "method": method,
                    "precision": round(m["precision"], 4),
                    "recall": round(m["recall"], 4),
                    "full_set_f1": round(m["full_set_f1"], 4),
                    "coverage": round(m["coverage"], 4),
                    "abstention": round(m["abstention"], 4),
                    "n_predicted": m["n_predicted"],
                    "status": m["status"],
                })
                print(f"{bridge:6s} {mask:20s} {method:16s} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['full_set_f1']:.4f} cov={m['coverage']:.4f} abst={m['abstention']:.4f} {m['status']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "masking_ladder.csv", index=False)
    (OUT / "masking_ladder.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    # markdown table
    lines = ["| bridge | mask | method | precision | recall | F1 | coverage | abstention | status |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        lines.append(f"| {r['bridge']} | {r['mask']} | {r['method']} | {r['precision']} | {r['recall']} | {r['full_set_f1']} | {r['coverage']} | {r['abstention']} | {r['status']} |")
    (OUT / "masking_ladder.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
