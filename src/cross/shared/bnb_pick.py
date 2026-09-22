from __future__ import annotations

import pandas as pd

from .normalize import norm_addr

ZERO = "0x0000000000000000000000000000000000000000"


def _numeric_value(series: pd.Series) -> pd.Series:
    v = series.astype(str).str.lower()

    def one(x: str) -> float:
        if x.startswith("0x"):
            try:
                return float(int(x, 16))
            except ValueError:
                return 0.0
        try:
            return float(x)
        except ValueError:
            return 0.0

    return v.map(one)


def pick_bnb_recipient(dst_rows: pd.DataFrame, address_a: str, bridge_address: str) -> tuple[str, str, int]:
    a = norm_addr(address_a)
    br = norm_addr(bridge_address)
    if dst_rows.empty:
        return "", "no_rows", 0
    df = dst_rows.copy()
    df["_val"] = _numeric_value(df["value"]) if "value" in df.columns else 0.0
    ca = df["contractAddress"].map(norm_addr) if "contractAddress" in df.columns else ""
    df["_ca"] = ca if isinstance(ca, pd.Series) else ""
    token_mask = df["_ca"].ne(ZERO) & df["_ca"].ne("")
    token_df = df[token_mask].copy()
    candidate_count = len(token_df)
    if not token_df.empty:
        to_l = token_df["to"].map(norm_addr)
        match = token_df[to_l == a]
        if len(match) == 1:
            return norm_addr(match.iloc[0]["to"]), "to_matches_eth_from", candidate_count
        if len(match) > 1:
            i = match["_val"].idxmax()
            return norm_addr(match.loc[i, "to"]), "to_matches_eth_from_max_val", candidate_count
        if "from" in token_df.columns:
            from_l = token_df["from"].map(norm_addr)
            mint = token_df[from_l == ZERO]
            if not mint.empty:
                mint = mint[mint["to"].map(norm_addr).ne(br)]
                if len(mint) == 1:
                    return norm_addr(mint.iloc[0]["to"]), "mint_single_non_bridge", candidate_count
                if len(mint) > 1:
                    uniq_to = mint["to"].map(norm_addr).unique()
                    if len(uniq_to) == 1:
                        return uniq_to[0], "mint_unique_recipient", candidate_count
                    i = mint["_val"].idxmax()
                    return norm_addr(mint.loc[i, "to"]), "mint_max_val", candidate_count
        non_br = token_df[token_df["to"].map(norm_addr).ne(br)]
        if not non_br.empty:
            i = non_br["_val"].idxmax()
            return norm_addr(non_br.loc[i, "to"]), "token_max_val_fallback", candidate_count
    df2 = df.copy()
    if "to" in df2.columns:
        to_l = df2["to"].map(norm_addr)
        hit = df2[to_l == a]
        if len(hit) == 1:
            return a, "native_or_fallback_to_matches_a", len(df2)
        non_br = df2[df2["to"].map(norm_addr).ne(br)]
        if not non_br.empty:
            i = non_br["_val"].idxmax()
            return norm_addr(non_br.loc[i, "to"]), "native_max_to_non_bridge", len(df2)
    return "", "unresolved", candidate_count
