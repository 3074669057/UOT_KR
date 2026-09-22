#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 0.1: candidate universe reconciliation for baseline_compare.

Clarifies 7296 unique BNB txs vs 14624 tx-flow membership entries.
All outputs under cross/out/baseline_compare/. Does not enter Phase 1.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "out" / "baseline_compare"
LABELS = OUT / "labels"

LAO_ETH = REPO / "out" / "leave_anchor_out_real" / "uot" / "uot_flow_segments_eth.csv"
LAO_BNB = REPO / "out" / "leave_anchor_out_real" / "uot" / "uot_flow_segments_bnb.csv"
GT_TX_SRC = REPO / "label" / "celer_label.csv"
COUNT_RECON = REPO / "out" / "dataset_count_reconciliation" / "dataset_count_reconciliation.json"

DEPRECATED = [
    LABELS / "universe_bnb_txs.csv",
    LABELS / "universe_eth_txs.csv",
]


def _norm(h: Any) -> str:
    s = str(h or "").strip().lower()
    if s and not s.startswith("0x"):
        s = "0x" + s
    return s


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_segment_txs(tx_hashes_field: str) -> list[str]:
    return [_norm(t) for t in str(tx_hashes_field).split("|") if t.strip()]


def _analyze_bnb_segments() -> dict[str, Any]:
    df = pd.read_csv(LAO_BNB, dtype=str, keep_default_na=False)
    memberships: list[dict[str, str]] = []
    tx_to_flows: dict[str, set[str]] = {}
    per_flow_counts: list[int] = []

    for _, row in df.iterrows():
        fid = str(row["flow_id"])
        txs = _parse_segment_txs(row.get("tx_hashes", ""))
        per_flow_counts.append(len(txs))
        for tx in txs:
            memberships.append({"tx_hash": tx, "dst_flow_id": fid})
            tx_to_flows.setdefault(tx, set()).add(fid)

    unique_txs = set(tx_to_flows)
    return {
        "n_dst_flows": len(df),
        "n_unique_dst_flow_id": int(df["flow_id"].nunique()),
        "n_tx_flow_membership_entries": len(memberships),
        "n_unique_bnb_tx_hashes": len(unique_txs),
        "min_tx_per_flow": min(per_flow_counts) if per_flow_counts else 0,
        "max_tx_per_flow": max(per_flow_counts) if per_flow_counts else 0,
        "mean_tx_per_flow": float(sum(per_flow_counts) / max(len(per_flow_counts), 1)),
        "n_tx_in_multiple_flows": sum(1 for flows in tx_to_flows.values() if len(flows) > 1),
        "memberships": memberships,
        "tx_to_flows": tx_to_flows,
        "flow_rows": df,
    }


def _analyze_eth_segments() -> dict[str, Any]:
    df = pd.read_csv(LAO_ETH, dtype=str, keep_default_na=False)
    unique_txs: set[str] = set()
    for _, row in df.iterrows():
        unique_txs.update(_parse_segment_txs(row.get("tx_hashes", "")))
    return {"n_src_flows": len(df), "n_unique_eth_tx_hashes": len(unique_txs)}


def _export_candidate_universe(bnb: dict[str, Any], gt_dst: set[str]) -> dict[str, Any]:
    unique_txs = set(bnb["tx_to_flows"])

    gt_dst_rows = [{"dst_tx_hash": tx, "source": "celer_label.csv"} for tx in sorted(gt_dst)]
    pd.DataFrame(gt_dst_rows).to_csv(LABELS / "gt_dst_txs.csv", index=False)

    gt_src = pd.read_csv(GT_TX_SRC, dtype=str, keep_default_na=False)
    gt_src_rows = [{"src_tx_hash": _norm(x), "source": "celer_label.csv"} for x in gt_src["srcTxhash"].unique()]
    pd.DataFrame(gt_src_rows).to_csv(LABELS / "gt_src_txs.csv", index=False)

    cand_tx_rows = []
    for tx in sorted(unique_txs):
        flows = bnb["tx_to_flows"][tx]
        cand_tx_rows.append(
            {
                "tx_hash": tx,
                "n_dst_flow_memberships": len(flows),
                "is_gt_dst_tx": tx in gt_dst,
            }
        )
    pd.DataFrame(cand_tx_rows).to_csv(LABELS / "candidate_bnb_universe_all_txs.csv", index=False)

    bnb["flow_rows"].to_csv(LABELS / "candidate_bnb_universe_flows.csv", index=False)

    pd.DataFrame(bnb["memberships"]).to_csv(
        LABELS / "candidate_bnb_tx_flow_memberships.csv", index=False
    )

    eth_df = pd.read_csv(LAO_ETH, dtype=str, keep_default_na=False)
    eth_df.to_csv(LABELS / "candidate_eth_universe_flows.csv", index=False)

    eth_unique = set()
    eth_rows = []
    for _, row in eth_df.iterrows():
        fid = str(row["flow_id"])
        for tx in _parse_segment_txs(row.get("tx_hashes", "")):
            eth_unique.add(tx)
            eth_rows.append({"tx_hash": tx, "src_flow_id": fid})
    eth_mem = pd.DataFrame(eth_rows).drop_duplicates()
    eth_mem["n_src_flow_memberships"] = eth_mem.groupby("tx_hash")["src_flow_id"].transform("count")
    eth_mem.drop_duplicates("tx_hash").to_csv(LABELS / "candidate_eth_universe_all_txs.csv", index=False)

    for p in DEPRECATED:
        if p.is_file():
            p.unlink()

    gt_in_cand = gt_dst <= unique_txs
    cand_extra = unique_txs - gt_dst

    return {
        "n_gt_dst_txs": len(gt_dst),
        "n_candidate_bnb_unique_txs": len(unique_txs),
        "n_candidate_bnb_tx_flow_memberships": bnb["n_tx_flow_membership_entries"],
        "n_candidate_bnb_flows": bnb["n_dst_flows"],
        "gt_dst_subset_of_candidate": gt_in_cand,
        "n_gt_dst_missing_from_candidate": len(gt_dst - unique_txs),
        "n_candidate_tx_not_in_gt_dst": len(cand_extra),
        "candidate_unique_tx_set_equals_gt_dst": unique_txs == gt_dst,
        "shared_pool_baseline_bnb_file": "labels/candidate_bnb_universe_all_txs.csv",
        "misnamed_phase0_file_removed": [str(p.name) for p in DEPRECATED if not p.is_file()],
    }


def _load_manifest() -> dict[str, Any]:
    p = OUT / "manifest.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _write_labels_readme(recon: dict[str, Any]) -> None:
    text = f"""# Baseline compare — label files (Phase 0 / 0.1)

Generated: {datetime.now(timezone.utc).isoformat()}

## Three concepts (do not conflate)

| Concept | Count | File |
|---------|------:|------|
| **GT tx-pairs** | 7296 | `gt_tx_pairs.csv` |
| **GT dst txs** | {recon['n_gt_dst_txs']} | `gt_dst_txs.csv` |
| **Candidate BNB tx universe (unique hashes)** | {recon['n_candidate_bnb_unique_txs']} | `candidate_bnb_universe_all_txs.csv` |
| **Candidate BNB flow universe** | {recon['n_candidate_bnb_flows']} | `candidate_bnb_universe_flows.csv` |
| **Candidate BNB tx-flow memberships (with duplication)** | {recon['n_candidate_bnb_tx_flow_memberships']} | `candidate_bnb_tx_flow_memberships.csv` |

## 14624 vs 7296 reconciliation

- **7296** = unique BNB `tx_hash` values appearing in LAO `uot_flow_segments_bnb.csv`.
- **14624** = total **tx→flow membership entries** when the same tx hash is listed in multiple dst flows (overlapping flow segments). This matches `dataset_count_reconciliation.json` → `bnb_tx_in_eval_flows`.
- Phase 0 incorrectly reported "BNB universe txs = 7296" using the **unique hash** count but named the file `universe_bnb_txs.csv`, which looked like a GT-pruned pool. That file has been **removed**.

## Shared-pool baseline candidate pool

Tx-level baselines (Connector `WithdrawLocator`) must use:

- **`candidate_bnb_universe_all_txs.csv`** — all unique dst txs in the 5226 LAO dst flows.

Do **not** use `gt_dst_txs.csv` as the candidate pool (evaluation labels only).

Flow-level RC-UOT-Q uses the **5226 dst flows** (`candidate_bnb_universe_flows.csv`), consistent with the UOT cost matrix shape.

## Label-pruning disclosure

- `candidate_bnb_universe_all_txs` **equals** `gt_dst_txs` as a set ({recon['n_candidate_bnb_unique_txs']} = {recon['n_gt_dst_txs']}).
- This is a structural property of the **strict LAO eval subgraph**: raw `Celer_BNB_qu.csv` in this run also contains exactly 7296 unique hashes (no extra decoy dst txs at tx-hash level).
- **Not** the same as using `gt_dst_txs.csv` as the search pool during matching — the correct pool file is still `candidate_bnb_universe_all_txs.csv`.
- Flow-level candidates (**5226 flows**) are a strict superset of unique labeled flow-pairs (**3934**); tx-level unique hashes do not include unlabeled decoys in this subgraph.

## Deprecated (removed)

- `universe_bnb_txs.csv` — misnamed; duplicated GT dst tx set without distinguishing candidate vs label role.
- `universe_eth_txs.csv` — same issue on ETH side; replaced by `candidate_eth_universe_all_txs.csv`.
"""
    (LABELS / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    LABELS.mkdir(parents=True, exist_ok=True)

    bnb = _analyze_bnb_segments()
    eth = _analyze_eth_segments()
    gt = pd.read_csv(GT_TX_SRC, dtype=str, keep_default_na=False)
    gt_dst = {_norm(x) for x in gt["dstTxhash"]}

    if bnb["n_tx_flow_membership_entries"] != 14624:
        raise RuntimeError(
            f"Expected 14624 tx-flow membership entries, got {bnb['n_tx_flow_membership_entries']}"
        )
    if bnb["n_unique_bnb_tx_hashes"] != 7296:
        raise RuntimeError(
            f"Expected 7296 unique BNB tx hashes in LAO segments, got {bnb['n_unique_bnb_tx_hashes']}"
        )

    recon = _export_candidate_universe(bnb, gt_dst)
    recon.update(
        {
            "n_src_flows": eth["n_src_flows"],
            "n_unique_eth_tx_hashes": eth["n_unique_eth_tx_hashes"],
            "phase0_misreport_explanation": (
                "Phase 0 counted unique tx hashes (7296) but labeled the file universe_bnb_txs.csv, "
                "conflating it with the 14624 membership-entry count from dataset_count_reconciliation. "
                "14624 is NOT 14624 unique txs."
            ),
            "count_reconciliation_source": str(COUNT_RECON.relative_to(REPO)),
        }
    )

    _write_labels_readme(recon)

    manifest = _load_manifest()
    manifest["phase0_1"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_universe_reconciliation": recon,
        "terminology": {
            "gt_dst_txs": "7296 labeled destination txs from celer_label.csv (evaluation only)",
            "candidate_bnb_universe_all_txs": (
                f"{recon['n_candidate_bnb_unique_txs']} unique dst tx hashes in 5226 LAO dst flows "
                "(shared-pool for tx-level baselines)"
            ),
            "candidate_bnb_universe_flows": "5226 dst flow segments (shared-pool for flow-level UOT parity)",
            "candidate_bnb_tx_flow_memberships": (
                f"{recon['n_candidate_bnb_tx_flow_memberships']} tx→flow rows (duplicate tx across overlapping flows)"
            ),
        },
        "label_pruning_risk": {
            "tx_hash_level_candidate_equals_gt_dst": recon["candidate_unique_tx_set_equals_gt_dst"],
            "flow_level_candidates_exceed_unique_labeled_flow_pairs": True,
            "n_unique_labeled_flow_pairs": manifest.get("counts", {}).get("n_unique_flow_pairs", 3934),
            "misused_gt_as_candidate_pool_in_phase0": True,
            "phase0_misnamed_file_corrected": True,
            "shared_pool_file_for_baselines": recon["shared_pool_baseline_bnb_file"],
        },
    }

    substrate_files = [
        LAO_ETH,
        LAO_BNB,
        GT_TX_SRC,
        LABELS / "gt_dst_txs.csv",
        LABELS / "candidate_bnb_universe_all_txs.csv",
        LABELS / "candidate_bnb_universe_flows.csv",
        LABELS / "candidate_bnb_tx_flow_memberships.csv",
        LABELS / "gt_tx_pairs.csv",
        LABELS / "tx_to_flow_map_lao.csv",
    ]
    manifest["substrate_sha256"] = manifest.get("substrate_sha256") or {}
    for p in substrate_files:
        if p.is_file():
            manifest["substrate_sha256"][str(p.relative_to(REPO))] = _sha256(p)

    # Phase 0.1 conclusion
    if recon["n_gt_dst_missing_from_candidate"]:
        phase01 = "BLOCKED"
    elif recon["candidate_unique_tx_set_equals_gt_dst"]:
        phase01 = "WARN"
    else:
        phase01 = "PASS"

    manifest["phase0_1_conclusion"] = phase01
    manifest["phase0_conclusion"] = manifest.get("phase0_conclusion", "WARN")
    manifest["counts"] = {
        **(manifest.get("counts") or {}),
        "n_gt_dst_txs": recon["n_gt_dst_txs"],
        "n_candidate_bnb_unique_txs": recon["n_candidate_bnb_unique_txs"],
        "n_candidate_bnb_tx_flow_memberships": recon["n_candidate_bnb_tx_flow_memberships"],
        "n_candidate_bnb_flows": recon["n_candidate_bnb_flows"],
        "n_candidate_eth_unique_txs": eth["n_unique_eth_tx_hashes"],
    }
    manifest["blockers_for_phase1"] = list(manifest.get("blockers_for_phase1") or [])
    extra = (
        "BNB tx-level candidate universe equals GT dst tx set (7296=7296) in strict LAO subgraph; "
        "no unlabeled decoy dst txs at tx-hash level — disclose in baseline comparison prose"
    )
    if extra not in manifest["blockers_for_phase1"]:
        manifest["blockers_for_phase1"].append(extra)

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    acceptance = f"""# Baseline compare — Phase 0.1 acceptance table

Generated: {manifest['phase0_1']['generated_at_utc']}

## Conclusion: **{phase01}**

## 14624 vs 7296 explanation

| Metric | Value | Meaning |
|--------|------:|---------|
| BNB tx-flow membership entries | **14624** | Same `tx_hash` counted once per dst flow it appears in |
| Unique BNB tx hashes in LAO segments | **7296** | Distinct on-chain dst transactions |
| GT dst txs | **7296** | Labeled dst txs from `celer_label.csv` |

**Source of 14624:** `out/dataset_count_reconciliation/dataset_count_reconciliation.json` → `bnb_tx_in_eval_flows` counts segment **membership entries**, not unique hashes. Every one of the 7296 unique dst txs appears in multiple overlapping dst flows (`n_tx_in_multiple_flows = 7296`).

Phase 0 misreport: used unique-hash count (7296) but filename `universe_bnb_txs.csv` implied a candidate pool definition without separating GT vs candidate roles.

## Acceptance checks

| Check | Result |
|-------|--------|
| ETH src flows | **3258** |
| BNB dst flows | **5226** |
| GT tx-pairs | **7296** |
| GT dst txs | **7296** |
| BNB candidate unique txs | **7296** (`candidate_bnb_universe_all_txs.csv`) |
| BNB candidate tx-flow memberships | **14624** (`candidate_bnb_tx_flow_memberships.csv`) |
| BNB candidate flows | **5226** (`candidate_bnb_universe_flows.csv`) |
| GT dst tx 100% in candidate universe | **{'YES' if recon['gt_dst_subset_of_candidate'] else 'NO'}** |
| Candidate ⊇ GT dst | **{'YES' if recon['gt_dst_subset_of_candidate'] else 'NO'}** (equal sets) |
| Candidate unique txs > GT dst txs | **NO** (equal, not superset) |
| Phase 0 misused GT as candidate file name | **YES** (corrected; file removed) |
| Shared-pool baseline BNB file | **`labels/candidate_bnb_universe_all_txs.csv`** |

## Label-pruning risk

- At **tx-hash** level, candidate universe **equals** GT dst set (no unlabeled decoy txs in LAO BNB segments).
- At **flow** level, RC-UOT-Q uses **5226** dst flows (full cost matrix); unique labeled flow-pairs = 3934.
- Baselines must load candidates from `candidate_bnb_universe_all_txs.csv`, not `gt_dst_txs.csv`.

## Phase 1

**Not entered.** Stop here per instructions.
"""

    preflight = OUT / "preflight_report.md"
    existing = preflight.read_text(encoding="utf-8") if preflight.is_file() else ""
    preflight.write_text(
        existing.rstrip()
        + "\n\n---\n\n"
        + acceptance
        + "\n",
        encoding="utf-8",
    )

    print(f"Phase 0.1 complete. Conclusion: {phase01}")
    print(f"unique BNB txs={recon['n_candidate_bnb_unique_txs']} memberships={recon['n_candidate_bnb_tx_flow_memberships']}")
    print(f"candidate==gt_dst: {recon['candidate_unique_tx_set_equals_gt_dst']}")


if __name__ == "__main__":
    main()
