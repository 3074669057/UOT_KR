"""Label CSV helpers: src hash → dst hash index."""
from __future__ import annotations

import pandas as pd

from cross.shared.normalize import norm_addr


def label_dst_by_src(labels: pd.DataFrame) -> dict[str, str]:
    """``norm(srcTxhash) -> norm(dstTxhash)``. Later rows overwrite earlier on duplicate src."""
    out: dict[str, str] = {}
    if labels is None or labels.empty:
        return out
    for _, r in labels.iterrows():
        s = norm_addr(r.get("srcTxhash", "") or r.get("srcTxHash", ""))
        d = norm_addr(r.get("dstTxhash", "") or r.get("dstTxHash", ""))
        if s and d:
            out[s] = d
    return out


__all__ = ["label_dst_by_src"]
