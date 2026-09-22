"""Coverage-precision curve for RC-UOT-Q vs baselines (three bridges, held-out).

RC-UOT-Q's solver config is frozen from the unmasked dev selection; only the
confidence threshold is swept to trace the abstention/coverage/precision curve.
Baselines are single operating points (they do not abstain by design).
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

from run_rc_uot_q_multi_bridge import load_bridge_cached, prepare_rc, truth_for  # noqa: E402
from cross.application.experiments import ec_uot_q_final  # noqa: E402
from cross.application.experiments.ec_uot_q_final import evaluate_full_set, predict  # noqa: E402

OUT = REPO / "out" / "multi_bridge_expansion" / "coverage_precision"
BRIDGES = ("Celer", "Multi", "Poly")
THRESHOLDS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for bridge in BRIDGES:
        cached = load_bridge_cached(bridge)
        sel = json.loads((REPO / "out" / "multi_bridge_expansion" / bridge / "rc_uot_q_selection.json").read_text(encoding="utf-8"))
        frozen = sel["selected_config"]
        split = cached["split"]
        source = cached["source"]
        test_source = source.loc[source["split"] == "test"]
        test_truth = truth_for(split, "test")
        src, cand = prepare_rc(test_source, cached["test_candidates"])
        for thr in THRESHOLDS:
            cfg = {**frozen, "confidence_threshold": thr}
            ec_uot_q_final._PREDICT_CACHE.clear()
            pred, _ = predict(src, cand, cfg)
            m = evaluate_full_set(test_truth, pred)
            rows.append({
                "bridge": bridge, "method": "RC-UOT-Q", "threshold": thr,
                "precision": round(m["precision"], 4), "recall": round(m["recall"], 4),
                "full_set_f1": round(m["full_set_f1"], 4), "coverage": round(m["coverage"], 4),
                "abstention": round(m["abstention"], 4), "n_predicted": m["n_predicted"],
            })
            print(f"{bridge:6s} RC-UOT-Q thr={thr:<4} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['full_set_f1']:.4f} cov={m['coverage']:.4f} abst={m['abstention']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "coverage_precision.csv", index=False)
    (OUT / "coverage_precision.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
