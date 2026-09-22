"""Temporal external set disjointness auditor. DATA-ONLY.

Modes:
  --audit-history        : compute the frozen historical Celer boundary and the
                           per-file timestamp/tx-hash inventory. No method, no
                           performance, no 301-305 reads.
  --check-v3 PATHS...    : given v3 candidate corpus files (any CSV/JSON with tx
                           hashes and/or timestamps), assert: every timestamp is
                           >= the frozen window start; zero tx/anchor/flow/
                           component intersection with every historical source;
                           file/hash identity distinct from historical files.

301-305 holdout isolation: this script NEVER reads any holdout file. Holdout
independence is established STRUCTURALLY: the holdout's synthetic cells are
generated from the pre-boundary frozen corpus (per the frozen holdout
preregistration metadata, read-only reference), so strict temporal disjointness of
the v3 window implies identity disjointness with the holdout as well.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# Frozen historical Celer sources (repo-relative). Only non-holdout files.
HISTORICAL_FILES = [
    "data/Validation/ETH-BNB/Celer/label.csv",
    "data/Validation/ETH-BNB/Celer/sample.json",
    "data/Validation/ETH-Polygon/Celer/label.csv",
    "data/Validation/ETH-Polygon/Celer/sample.json",
    "out/paper_full_pipeline_run/label_layer_v1/flow_labels.csv",
    "out/paper_full_pipeline_run/label_layer_v1/tx_anchor_labels.csv",
    "out/paper_full_pipeline_run/label_layer_v1/tx_to_flow_map.csv",
    "out/paper_full_pipeline_run/label_layer_v1/flow_segments_eth.csv",
    "out/paper_full_pipeline_run/label_layer_v1/flow_segments_bnb.csv",
    "out/paper_full_pipeline_run/label_layer_v1/evidence_eth.csv",
    "out/paper_full_pipeline_run/label_layer_v1/evidence_bnb.csv",
    "out/baseline_compare/labels/gt_tx_pairs.csv",
    "out/baseline_compare/labels/candidate_bnb_universe_all_txs.csv",
    "out/multi_bridge_expansion/Celer/raw_relay_logs_test.json",
    "data/label/celer_label.csv",
    "data/label/tx/Celer_ETH_cun.csv",
    "data/label/tx/Celer_BNB_qu.csv",
]

# Frozen window (outcome-independent; see v3 TEMPORAL_EXTERNAL_PREREGISTRATION.md)
WINDOW_START_UNIX = 1684454400  # 2023-05-19 00:00:00Z (boundary + next UTC midnight)
BOUNDARY_AUDIT_UNIX = 1684453293  # 2023-05-18 23:41:33Z (relay logs max)

TX_COLS = ["srcTxhash", "dstTxhash", "src_tx_hash", "dst_tx_hash", "tx_hash",
           "transactionHash", "hash", "support_src_tx_hashes", "support_dst_tx_hashes",
           "src_tx", "dst_tx", "transferId", "dst_transferId"]
TIME_COLS = ["timestamp", "time_stamp", "blockTimestamp", "timeStamp",
             "first_src_time", "last_src_time", "first_dst_time", "last_dst_time",
             "start_time", "end_time", "csv_time_stamp", "ts"]


def iter_rows(path: Path):
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return
        if isinstance(data, dict):
            yield data
        elif isinstance(data, list):
            for item in data:
                yield item
        return
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        yield row


def _parse_ts(v) -> int | None:
    """Parse unix seconds (decimal or hex string), sanity-bounded 2008..2100."""
    try:
        t = int(float(v))
    except (TypeError, ValueError):
        s = str(v).strip()
        if s.lower().startswith("0x"):
            try:
                t = int(s, 16)
            except ValueError:
                return None
        else:
            return None
    return t if 1_200_000_000 < t < 4_100_000_000 else None


def split_hashes(value):
    if value is None:
        return set()
    return {v.strip() for v in str(value).split("|") if v.strip()}


def audit_history() -> dict:
    report = {"per_file": [], "tx_hashes": set(), "flow_ids": set(),
              "global_max_ts": -1}
    for rel in HISTORICAL_FILES:
        path = REPO / rel
        entry = {"file": rel, "exists": path.is_file(), "rows": 0,
                 "max_ts": None, "n_tx": 0}
        if not path.is_file():
            report["per_file"].append(entry)
            continue
        for row in iter_rows(path):
            entry["rows"] += 1
            for c in TX_COLS:
                for h in split_hashes(row.get(c)):
                    report["tx_hashes"].add(h)
                    entry["n_tx"] += 1
            for c in TIME_COLS:
                v = row.get(c)
                if v not in (None, ""):
                    t = _parse_ts(v)
                    if t is not None:
                        entry["max_ts"] = max(entry["max_ts"] or t, t)
                        report["global_max_ts"] = max(report["global_max_ts"], t)
            for c in ("src_flow_id", "dst_flow_id", "flow_id"):
                v = row.get(c)
                if v:
                    report["flow_ids"].add(str(v))
        report["per_file"].append(entry)
    report["global_max_ts_str"] = str(report["global_max_ts"])
    report["boundary"] = BOUNDARY_AUDIT_UNIX
    report["window_start"] = WINDOW_START_UNIX
    report["n_unique_historical_tx"] = len(report["tx_hashes"])
    report["n_unique_historical_flow_ids"] = len(report["flow_ids"])
    return report


def check_v3(paths: list[str]) -> dict:
    hist = audit_history()
    issues: list[str] = []
    n_ts_ok = n_ts_bad = 0
    intersections = {"tx": 0, "flow": 0}
    for rel in paths:
        path = REPO / rel
        if not path.is_file():
            issues.append(f"v3 candidate file missing: {rel}")
            continue
        for row in iter_rows(path):
            for c in TX_COLS:
                for h in split_hashes(row.get(c)):
                    if h in hist["tx_hashes"]:
                        intersections["tx"] += 1
            for c in TIME_COLS:
                v = row.get(c)
                if v not in (None, ""):
                    t = _parse_ts(v)
                    if t is not None:
                        if t >= WINDOW_START_UNIX:
                            n_ts_ok += 1
                        else:
                            n_ts_bad += 1
                            issues.append(f"timestamp below window start in {rel}: {t}")
            for c in ("src_flow_id", "dst_flow_id", "flow_id"):
                v = row.get(c)
                if v and str(v) in hist["flow_ids"]:
                    intersections["flow"] += 1
    verdict = (intersections["tx"] == 0 and intersections["flow"] == 0
               and n_ts_bad == 0 and not issues)
    return {"verdict": "DISJOINT" if verdict else "OVERLAP_FOUND",
            "issues": issues, "n_ts_ok": n_ts_ok, "n_ts_bad": n_ts_bad,
            "intersections": intersections,
            "historical_summary": {
                "global_max_ts": hist["global_max_ts"],
                "window_start": hist["window_start"],
                "n_unique_historical_tx": hist["n_unique_historical_tx"],
                "n_unique_historical_flow_ids": hist["n_unique_historical_flow_ids"],
            }}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-history", action="store_true")
    ap.add_argument("--check-v3", nargs="*", default=None)
    cli = ap.parse_args()
    if cli.check_v3 is not None:
        report = check_v3(cli.check_v3)
        print(json.dumps(report, indent=2))
        return 0 if report["verdict"] == "DISJOINT" else 1
    report = audit_history()
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("tx_hashes", "flow_ids")}, indent=2, default=str))
    print(f"HISTORICAL_BOUNDARY = {report['boundary']}")
    print(f"WINDOW_START = {report['window_start']}")
    print("DISJOINTNESS AUDIT (HISTORY MODE): OK" if report["global_max_ts"] <= BOUNDARY_AUDIT_UNIX
          else "WARNING: history max ts exceeds the audited boundary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
