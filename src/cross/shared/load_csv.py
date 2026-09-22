"""Load label and bridge CSV exports."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .normalize import norm_addr


def load_label_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = df[[c for c in ["srcnet", "srcTxhash", "dstnet", "dstTxhash"] if c in df.columns]]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    return df


def load_eth_cun(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df["hash"] = df["hash"].astype(str).str.strip().str.lower()
    return df


def bridge_address_from_eth_df(eth_df: pd.DataFrame) -> str:
    if "Address" not in eth_df.columns:
        return ""
    for x in eth_df["Address"]:
        s = str(x).strip()
        if s and s.lower() != "nan":
            return norm_addr(s)
    return ""
