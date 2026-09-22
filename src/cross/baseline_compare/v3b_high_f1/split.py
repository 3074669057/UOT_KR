"""Deterministic dev/test split for v3b."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cross.baseline_compare.v3b_high_f1.constants import DEV_RATIO, HASH_SEED, TEST_RATIO
from cross.shared.normalize import norm_addr


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def split_bucket(src_tx: str) -> str:
    h = hashlib.sha256(f"{HASH_SEED}:{norm_addr(src_tx)}".encode()).hexdigest()
    val = int(h[:8], 16) / 0xFFFFFFFF
    return "dev" if val < DEV_RATIO else "test"


def make_split(*, labels_dir: Path, output_dir: Path) -> dict[str, Any]:
    gt = pd.read_csv(labels_dir / "gt_tx_pairs.csv", dtype=str)
    gt["src_tx_hash"] = gt["src_tx_hash"].astype(str).map(norm_addr)
    gt["dst_tx_hash"] = gt["dst_tx_hash"].astype(str).map(norm_addr)
    gt["split"] = gt["src_tx_hash"].map(split_bucket)

    dev = gt[gt["split"] == "dev"].copy()
    test = gt[gt["split"] == "test"].copy()

    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    dev_path = split_dir / "dev_pairs.csv"
    test_path = split_dir / "test_pairs.csv"
    dev.to_csv(dev_path, index=False)
    test.to_csv(test_path, index=False)

    manifest = {
        "generated_at_utc": _utc(),
        "hash_seed": HASH_SEED,
        "hash_method": "sha256(f'{HASH_SEED}:{src_tx_hash}') -> first 8 hex / 2^32",
        "dev_ratio": DEV_RATIO,
        "test_ratio": TEST_RATIO,
        "n_total_pairs": len(gt),
        "n_dev_pairs": len(dev),
        "n_test_pairs": len(test),
        "n_dev_unique_src": int(dev["src_tx_hash"].nunique()),
        "n_test_unique_src": int(test["src_tx_hash"].nunique()),
        "dev_pairs_path": str(dev_path.relative_to(output_dir)).replace("\\", "/"),
        "test_pairs_path": str(test_path.relative_to(output_dir)).replace("\\", "/"),
    }
    _write_json(split_dir / "split_manifest.json", manifest)
    return manifest


def load_split_pairs(output_dir: Path, split: str) -> pd.DataFrame:
    path = output_dir / "splits" / f"{split}_pairs.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Split not found: {path}. Run --make-split first.")
    return pd.read_csv(path, dtype=str)


def truth_dict(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        s = norm_addr(r["src_tx_hash"])
        d = norm_addr(r["dst_tx_hash"])
        if s:
            out[s] = d
    return out
