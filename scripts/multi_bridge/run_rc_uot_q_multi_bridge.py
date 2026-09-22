from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cross.application.experiments.ec_uot_q_final import evaluate_full_set, predict, validate_inference_frame

OUT_ROOT = REPO / "out" / "multi_bridge_expansion"
BRIDGES = ("Celer", "Multi", "Poly")

SOURCE_ALLOWLIST = [
    "source_tx_hash",
    "source_timestamp",
    "source_amount_raw",
    "source_token_address",
    "source_route_type",
]
TARGET_ALLOWLIST = [
    "candidate_tx_hash",
    "candidate_timestamp",
    "amount_raw",
    "token_address",
    "contract_address",
    "topic0",
    "route_type",
    "bridge_contract_hit",
]

WINDOWS_H = [1.0, 3.0, 6.0, 24.0]
THRESHOLDS = [0.0, 0.5, 0.75]
RC_PRESETS = {
    "amount_time": {
        "amount_weight": 0.75,
        "time_weight": 0.25,
        "token_weight": 0.0,
        "route_weight": 0.0,
    },
    "token_aware": {
        "amount_weight": 0.50,
        "time_weight": 0.25,
        "token_weight": 0.25,
        "route_weight": 0.0,
    },
}
TRANSPORT = {"reg": 0.05, "reg_m": 0.5, "max_iter": 100, "tol": 1e-6, "top_k": 20, "batch_size": 50}
BEFORE_SEC = 900.0


def norm(value: object) -> str:
    return str(value or "").strip().lower()


def _as_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def _as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def load_old_module():
    path = REPO / "scripts" / "multi_bridge" / "run_eth_bnb_expansion.py"
    spec = importlib.util.spec_from_file_location("_multi_bridge_old", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OLD = load_old_module()


def load_bridge_cached(bridge: str) -> dict[str, Any]:
    root = OUT_ROOT / bridge
    split = pd.read_csv(root / "chronological_split_assignments.csv", dtype=str).fillna("")
    source = pd.read_csv(root / "source_features.csv")
    dev_candidates = pd.read_csv(root / "development_candidates.csv")
    test_candidates = pd.read_csv(root / "test_candidates.csv")

    receiver: dict[str, str] = {}
    sample_path = REPO / "data" / "Validation" / "ETH-BNB" / bridge / "sample.json"
    if sample_path.is_file():
        samples = json.loads(sample_path.read_text(encoding="utf-8"))
        for item in samples:
            src = norm(item.get("txhash"))
            recv = norm((item.get("args") or {}).get("receiver"))
            if src and recv:
                receiver[src] = recv
    return {
        "root": root,
        "split": split,
        "source": source,
        "dev_candidates": dev_candidates,
        "test_candidates": test_candidates,
        "receiver": receiver,
    }


def truth_for(split: pd.DataFrame, split_name: str) -> dict[str, str]:
    rows = split.loc[split["split"] == split_name]
    return {norm(r["source_tx_hash"]): norm(r["dest_tx_hash"]) for _, r in rows.iterrows()}


def prepare_rc(source: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = source.copy()
    for col in SOURCE_ALLOWLIST:
        if col not in src.columns:
            src[col] = "" if col not in {"source_timestamp", "source_amount_raw"} else (0 if col == "source_timestamp" else 0.0)
    src["source_tx_hash"] = src["source_tx_hash"].map(norm)
    src["source_timestamp"] = _as_int(src["source_timestamp"])
    src["source_amount_raw"] = _as_float(src["source_amount_raw"])
    src["source_token_address"] = src["source_token_address"].map(norm)
    if "source_route_type" not in src.columns or src["source_route_type"].isna().all():
        src["source_route_type"] = "bridge"
    src["source_route_type"] = src["source_route_type"].fillna("bridge").astype(str).str.lower()
    src = src[SOURCE_ALLOWLIST]

    cand = candidates.copy()
    for col in TARGET_ALLOWLIST:
        if col not in cand.columns:
            cand[col] = "" if col not in {"candidate_timestamp", "amount_raw", "bridge_contract_hit"} else (0 if col == "candidate_timestamp" else False)
    cand["candidate_tx_hash"] = cand["candidate_tx_hash"].map(norm)
    cand["candidate_timestamp"] = _as_int(cand["candidate_timestamp"])
    cand["amount_raw"] = _as_float(cand["amount_raw"])
    cand["token_address"] = cand["token_address"].map(norm)
    cand["contract_address"] = cand["contract_address"].map(norm)
    cand["topic0"] = cand["topic0"].map(norm)
    cand["route_type"] = cand["route_type"].fillna("bridge_event").astype(str).str.lower()
    cand["bridge_contract_hit"] = cand["bridge_contract_hit"].astype(bool)
    cand = cand[TARGET_ALLOWLIST]
    validate_inference_frame(src, side="source")
    validate_inference_frame(cand, side="target")
    return src, cand


def prepare_baseline_source(source: pd.DataFrame, receiver: dict[str, str]) -> pd.DataFrame:
    out = source.copy()
    out["source_tx_hash"] = out["source_tx_hash"].map(norm)
    out["source_receiver"] = out["source_tx_hash"].map(lambda h: receiver.get(h, ""))
    out["source_timestamp"] = _as_int(out["source_timestamp"])
    out["source_amount_raw"] = _as_float(out["source_amount_raw"])
    out["source_token_address"] = out["source_token_address"].map(norm)
    return out


def prepare_baseline_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    out["candidate_tx_hash"] = out["candidate_tx_hash"].map(norm)
    out["candidate_timestamp"] = _as_int(out["candidate_timestamp"])
    out["amount_raw"] = _as_float(out["amount_raw"])
    out["token_address"] = out["token_address"].map(norm)
    out["receiver"] = out["receiver"].fillna("").map(norm)
    return out


def normalize_baseline_amounts(source: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eth_dec, bnb_dec = OLD.load_token_decimals()
    return OLD.add_normalized_amounts(source, candidates, eth_dec, bnb_dec)


def candidate_pool_recall(truth: dict[str, str], candidates: pd.DataFrame) -> float:
    ids = set(candidates["candidate_tx_hash"].map(norm))
    covered = sum(norm(dst) in ids for dst in truth.values())
    return covered / max(len(truth), 1)


def metrics_with_config(truth: dict[str, str], pred: dict[str, str | None], cfg: dict[str, Any]) -> dict[str, Any]:
    metrics = evaluate_full_set(truth, pred)
    metrics.update({"status": "OK", "config": cfg})
    return metrics


def run_rc_uot_q(bridge: str, cached: dict[str, Any]) -> dict[str, Any]:
    root = cached["root"]
    split = cached["split"]
    source = cached["source"]
    dev_truth = truth_for(split, "development")
    test_truth = truth_for(split, "test")
    dev_source = source.loc[source["split"] == "development"]
    test_source = source.loc[source["split"] == "test"]
    src_dev, cand_dev = prepare_rc(dev_source, cached["dev_candidates"])
    src_test, cand_test = prepare_rc(test_source, cached["test_candidates"])

    rows: list[dict[str, Any]] = []
    for window_h in WINDOWS_H:
        for preset_name, preset in RC_PRESETS.items():
            for threshold in THRESHOLDS:
                cfg = {
                    "window_h": window_h,
                    "before_sec": BEFORE_SEC,
                    "weight_preset": preset_name,
                    "confidence_threshold": threshold,
                    "grouping_strategy": "token_30min",
                    "use_transport": True,
                    **preset,
                    **TRANSPORT,
                }
                pred, _ = predict(src_dev, cand_dev, cfg)
                metrics = evaluate_full_set(dev_truth, pred)
                rows.append(
                    {
                        "window_h": window_h,
                        "weight_preset": preset_name,
                        "threshold": threshold,
                        "grouping_strategy": "token_30min",
                        **metrics,
                    }
                )

    dev_df = pd.DataFrame(rows)
    dev_df.to_csv(root / "rc_uot_q_development_grid.csv", index=False)
    eligible = dev_df.loc[dev_df["precision"] >= 0.90]
    selected_row = (
        eligible.sort_values(["full_set_f1", "coverage"], ascending=False).iloc[0]
        if not eligible.empty
        else dev_df.sort_values(["full_set_f1", "coverage"], ascending=False).iloc[0]
    )
    selected_cfg = {
        "window_h": float(selected_row["window_h"]),
        "before_sec": BEFORE_SEC,
        "weight_preset": str(selected_row["weight_preset"]),
        "confidence_threshold": float(selected_row["threshold"]),
        "grouping_strategy": "token_30min",
        "use_transport": True,
        **RC_PRESETS[str(selected_row["weight_preset"])],
        **TRANSPORT,
    }
    pred_test, meta_test = predict(src_test, cand_test, selected_cfg)
    test_metrics = evaluate_full_set(test_truth, pred_test)
    test_metrics.update(
        {
            "candidate_pool_recall": candidate_pool_recall(test_truth, cand_test),
            "selected_config": selected_cfg,
            "development_selection": selected_row.to_dict(),
        }
    )
    selection = {
        "bridge": bridge,
        "n_development": int(len(dev_truth)),
        "n_test": int(len(test_truth)),
        "development_candidate_count": int(len(cand_dev)),
        "test_candidate_count": int(len(cand_test)),
        "development_selection": selected_row.to_dict(),
        "selected_config": selected_cfg,
        "test_metrics": test_metrics,
    }
    (root / "rc_uot_q_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (root / "rc_uot_q_test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return {"method": "RC-UOT-Q", "bridge": bridge, **test_metrics}


def run_connector_style(bridge: str, cached: dict[str, Any]) -> dict[str, Any]:
    root = cached["root"]
    split = cached["split"]
    source = cached["source"]
    receiver = cached["receiver"]
    dev_truth = truth_for(split, "development")
    test_truth = truth_for(split, "test")

    source = prepare_baseline_source(source, receiver)
    dev_candidates = prepare_baseline_candidates(cached["dev_candidates"])
    test_candidates = prepare_baseline_candidates(cached["test_candidates"])
    all_candidates = pd.concat([dev_candidates, test_candidates], ignore_index=True).drop_duplicates(
        "candidate_tx_hash"
    )
    token_map = OLD.infer_token_map(split, all_candidates)

    dev_source = source.loc[source["split"] == "development"].copy()
    test_source = source.loc[source["split"] == "test"].copy()
    dev_source, dev_candidates = normalize_baseline_amounts(dev_source, dev_candidates)
    test_source, test_candidates = normalize_baseline_amounts(test_source, test_candidates)

    rows: list[dict[str, Any]] = []
    for window_h in WINDOWS_H:
        for threshold in THRESHOLDS:
            cfg = OLD.base_config(threshold)
            cfg["window_h"] = window_h
            pred, conf = OLD.predict_exact(dev_source, dev_candidates, token_map, cfg)
            pred = {
                src: (dst if dst is not None and conf.get(src, 0.0) >= threshold else None)
                for src, dst in pred.items()
            }
            metrics = evaluate_full_set(dev_truth, pred)
            rows.append({"window_h": window_h, "threshold": threshold, **metrics})

    dev_df = pd.DataFrame(rows)
    dev_df.to_csv(root / "connector_development_grid.csv", index=False)
    eligible = dev_df.loc[dev_df["precision"] >= 0.90]
    selected_row = (
        eligible.sort_values(["full_set_f1", "coverage"], ascending=False).iloc[0]
        if not eligible.empty
        else dev_df.sort_values(["full_set_f1", "coverage"], ascending=False).iloc[0]
    )
    selected_cfg = {
        "window_h": float(selected_row["window_h"]),
        "confidence_threshold": float(selected_row["threshold"]),
    }
    test_cfg = OLD.base_config(float(selected_row["threshold"]))
    test_cfg["window_h"] = float(selected_row["window_h"])
    pred_test, conf_test = OLD.predict_exact(test_source, test_candidates, token_map, test_cfg)
    pred_test = {
        src: (dst if dst is not None and conf_test.get(src, 0.0) >= float(selected_row["threshold"]) else None)
        for src, dst in pred_test.items()
    }
    test_metrics = evaluate_full_set(test_truth, pred_test)
    test_metrics.update(
        {
            "candidate_pool_recall": candidate_pool_recall(test_truth, test_candidates),
            "selected_config": selected_cfg,
            "development_selection": selected_row.to_dict(),
            "token_map_size": int(len(token_map)),
        }
    )
    selection = {
        "bridge": bridge,
        "n_development": int(len(dev_truth)),
        "n_test": int(len(test_truth)),
        "development_selection": selected_row.to_dict(),
        "selected_config": selected_cfg,
        "test_metrics": test_metrics,
    }
    (root / "connector_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (root / "connector_test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return {"method": "Connector-style adapted", "bridge": bridge, **test_metrics}


def predict_abctracer_style(source: pd.DataFrame, candidates: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, str | None], dict[str, float]]:
    before = float(config.get("before_sec", BEFORE_SEC))
    after = float(config["window_h"]) * 3600.0
    src = source.copy()
    cand = candidates.copy()
    for frame, time_col, amt_col, recv_col in (
        (src, "source_timestamp", "source_amount_human", "source_receiver"),
        (cand, "candidate_timestamp", "amount_human", "receiver"),
    ):
        frame[time_col] = _as_int(frame[time_col])
        frame[amt_col] = _as_float(frame[amt_col])
        frame[recv_col] = frame[recv_col].fillna("").map(norm)

    cand = cand.sort_values("candidate_timestamp").reset_index(drop=True)
    cand_ts = cand["candidate_timestamp"].to_numpy(dtype=float)
    cand_hashes = cand["candidate_tx_hash"].to_numpy(dtype=str)
    cand_amt = cand["amount_human"].to_numpy(dtype=float)
    cand_recv = cand["receiver"].to_numpy(dtype=str)

    predictions: dict[str, str | None] = {}
    confidence: dict[str, float] = {}
    for row in src.itertuples(index=False):
        src_h = str(row.source_tx_hash)
        ts0 = float(row.source_timestamp)
        amt0 = float(row.source_amount_human)
        recv0 = str(row.source_receiver or "")
        lo = int(np.searchsorted(cand_ts, ts0 - before, side="left"))
        hi = int(np.searchsorted(cand_ts, ts0 + after, side="right"))
        if lo >= hi:
            predictions[src_h] = None
            confidence[src_h] = 0.0
            continue
        sub_amt = cand_amt[lo:hi]
        sub_ts = cand_ts[lo:hi]
        sub_recv = cand_recv[lo:hi]
        amount_sim = 1.0 - np.minimum(
            np.abs(sub_amt - amt0) / np.maximum(np.maximum(np.abs(sub_amt), np.abs(amt0)), 1e-12), 1.0
        )
        delay = sub_ts - ts0
        time_sim = np.where(delay < 0, 0.0, 1.0 - np.minimum(delay / max(after, 1.0), 1.0))
        receiver_sim = ((sub_recv == recv0) & (recv0 != "")).astype(float)
        score = 0.55 * amount_sim + 0.25 * time_sim + 0.20 * receiver_sim
        best = int(np.argmax(score))
        if score[best] <= 0.0:
            predictions[src_h] = None
            confidence[src_h] = 0.0
        else:
            predictions[src_h] = str(cand_hashes[lo + best])
            confidence[src_h] = float(score[best])
    return predictions, confidence


def run_abctracer_style(bridge: str, cached: dict[str, Any]) -> dict[str, Any]:
    root = cached["root"]
    split = cached["split"]
    source = cached["source"]
    receiver = cached["receiver"]
    dev_truth = truth_for(split, "development")
    test_truth = truth_for(split, "test")

    source = prepare_baseline_source(source, receiver)
    dev_candidates = prepare_baseline_candidates(cached["dev_candidates"])
    test_candidates = prepare_baseline_candidates(cached["test_candidates"])

    dev_source = source.loc[source["split"] == "development"].copy()
    test_source = source.loc[source["split"] == "test"].copy()
    dev_source, dev_candidates = normalize_baseline_amounts(dev_source, dev_candidates)
    test_source, test_candidates = normalize_baseline_amounts(test_source, test_candidates)

    rows: list[dict[str, Any]] = []
    for window_h in WINDOWS_H:
        cfg = {"window_h": window_h, "before_sec": BEFORE_SEC}
        pred, _ = predict_abctracer_style(dev_source, dev_candidates, cfg)
        metrics = evaluate_full_set(dev_truth, pred)
        rows.append({"window_h": window_h, **metrics})

    dev_df = pd.DataFrame(rows)
    dev_df.to_csv(root / "abctracer_development_grid.csv", index=False)
    selected_row = dev_df.sort_values(["full_set_f1", "coverage"], ascending=False).iloc[0]
    selected_cfg = {"window_h": float(selected_row["window_h"]), "before_sec": BEFORE_SEC}
    pred_test, _ = predict_abctracer_style(test_source, test_candidates, selected_cfg)
    test_metrics = evaluate_full_set(test_truth, pred_test)
    test_metrics.update(
        {
            "candidate_pool_recall": candidate_pool_recall(test_truth, test_candidates),
            "selected_config": selected_cfg,
            "development_selection": selected_row.to_dict(),
        }
    )
    selection = {
        "bridge": bridge,
        "n_development": int(len(dev_truth)),
        "n_test": int(len(test_truth)),
        "development_selection": selected_row.to_dict(),
        "selected_config": selected_cfg,
        "test_metrics": test_metrics,
    }
    (root / "abctracer_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (root / "abctracer_test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return {"method": "ABCTracer-style adapted", "bridge": bridge, **test_metrics}


def write_summary(results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "bridge": r["bridge"],
                "method": r["method"],
                "n_total": r.get("n_total"),
                "n_predicted": r.get("n_predicted"),
                "precision": r.get("precision"),
                "recall": r.get("recall"),
                "full_set_f1": r.get("full_set_f1"),
                "coverage": r.get("coverage"),
                "abstention": r.get("abstention"),
                "candidate_pool_recall": r.get("candidate_pool_recall"),
                "window_h": r.get("selected_config", {}).get("window_h"),
                "threshold": r.get("selected_config", {}).get("confidence_threshold"),
                "weight_preset": r.get("selected_config", {}).get("weight_preset"),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_ROOT / "summary.csv", index=False)

    lines = [
        "| bridge | method | pairs | candidate recall | precision | recall | F1 | coverage | abstention | window | threshold | preset |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        fmt = lambda v, d=4: "NA" if v is None or pd.isna(v) else f"{float(v):.{d}f}"
        lines.append(
            f"| {r['bridge']} | {r['method']} | {int(r['n_total'])} | {fmt(r['candidate_pool_recall'])} | {fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['full_set_f1'])} | {fmt(r['coverage'])} | {fmt(r['abstention'])} | {r['window_h']} | {r['threshold']} | {r['weight_preset']} |"
        )
    (OUT_ROOT / "paper_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_ROOT / "paper_comparison.csv").write_text(
        pd.DataFrame(rows).to_csv(index=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", choices=BRIDGES, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    for bridge in ([args.bridge] if args.bridge else list(BRIDGES)):
        cached = load_bridge_cached(bridge)
        results.append(run_rc_uot_q(bridge, cached))
        results.append(run_connector_style(bridge, cached))
        results.append(run_abctracer_style(bridge, cached))
    write_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
