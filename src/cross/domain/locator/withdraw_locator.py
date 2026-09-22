# Vendored from Connector ``core/dst_chain.py`` (WithdrawLocator) for a standalone ``cross`` package.
# Token CSV paths use :func:`token_data_dir` instead of ``config.Config``.
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from ...config.paths import token_data_dir

_CHAIN_CSV = {
    "ETH": "ERC20.csv",
    "BNB": "BERC20.csv",
    "Polygon": "PERC20.csv",
}


def _token_csv_path_for_chain(chain: str) -> str | None:
    name = _CHAIN_CSV.get(chain)
    if not name:
        return None
    p: Path = token_data_dir() / name
    return str(p) if p.is_file() else None


def _collect_wanted_token_addresses(chains: set, src_txs: pd.DataFrame, dst_txs: pd.DataFrame) -> dict:
    """Addresses that need decimals from TOKEN_FILE per chain (avoids loading huge ERC20.csv)."""
    wanted = {c: set() for c in chains}
    if "args.srcChain" in src_txs.columns and "args.asset_s" in src_txs.columns:
        for _, row in src_txs.iterrows():
            sc = row.get("args.srcChain")
            if sc not in wanted:
                continue
            a = str(row.get("args.asset_s", "") or "").strip().lower()
            if a and a != "nan":
                wanted[sc].add(a)

    if dst_txs is None or dst_txs.empty or "contractAddress" not in dst_txs.columns:
        return wanted

    if "Net" in dst_txs.columns:
        for _, row in dst_txs.iterrows():
            net = str(row.get("Net", "") or "").strip()
            if net not in wanted:
                continue
            ca = str(row.get("contractAddress", "") or "").strip().lower()
            if ca and ca != "nan":
                wanted[net].add(ca)
    elif "args.dstChain" in src_txs.columns:
        fallback_dst = set(src_txs["args.dstChain"].dropna().astype(str)) & chains
        for dc in fallback_dst:
            for _, row in dst_txs.iterrows():
                ca = str(row.get("contractAddress", "") or "").strip().lower()
                if ca and ca != "nan":
                    wanted[dc].add(ca)
    return wanted


def _load_token_decimals_subset(csv_path: str, wanted: set) -> dict:
    """Load decimals only for wanted addresses via chunked scan (bounded memory)."""
    if not wanted:
        return {}
    need = {str(w).strip().lower() for w in wanted if w and str(w).strip().lower() not in ("", "nan")}
    if not need:
        return {}
    out = {}
    for chunk in pd.read_csv(
        csv_path,
        usecols=["address", "decimal"],
        chunksize=80000,
        dtype={"address": str},
    ):
        chunk["address"] = chunk["address"].str.lower()
        hit = chunk[chunk["address"].isin(need)]
        for addr, dec in zip(hit["address"], hit["decimal"]):
            out[addr] = int(dec)
        if need.issubset(out):
            break
    return out


class WithdrawLocator:
    """Withdrawal transaction pairing (cross-join + rule filters)."""

    def __init__(self, src_txs: pd.DataFrame, dst_txs: pd.DataFrame) -> None:
        self.src_txs = src_txs
        self.dst_txs = dst_txs
        self.src_tx_group = src_txs.groupby(["args.srcChain", "args.dstChain"])

        chains = set()
        for group in self.src_tx_group:
            src_chain, dst_chain = group[0]
            chains.add(src_chain)
            chains.add(dst_chain)

        wanted_by_chain = _collect_wanted_token_addresses(chains, src_txs, dst_txs)
        self.decimal_dict = {}
        for chain in chains:
            path = _token_csv_path_for_chain(chain)
            if not path:
                self.decimal_dict[chain] = {}
                continue
            wanted = wanted_by_chain.get(chain, set())
            self.decimal_dict[chain] = _load_token_decimals_subset(path, wanted)

    def _match_receiver(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["args.receiver"].str.lower() == df["to"].str.lower()].copy()

    def _match_tx_type(self, df: pd.DataFrame) -> pd.DataFrame:
        df_1 = df[(df["args.asset_s"] == "") & (df["contractAddress"] == "")].copy()
        df_2 = df[(df["args.asset_s"] != "") & (df["contractAddress"] != "")].copy()
        return pd.concat([df_1, df_2], axis=0)

    def _match_amount(self, df: pd.DataFrame, src_chain: str, dst_chain: str, threshold: float) -> pd.DataFrame:
        df.loc[:, "value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
        df.loc[:, "args.amount"] = pd.to_numeric(df["args.amount"], errors="coerce").fillna(0)

        df.loc[:, "value"] = df.apply(
            lambda x: x["value"]
            / pow(10, self.decimal_dict[dst_chain][str(x["contractAddress"]).lower()])
            if str(x["contractAddress"]).lower() in self.decimal_dict[dst_chain]
            else x["value"],
            axis=1,
        )

        df.loc[:, "args.amount"] = df.apply(
            lambda x: x["args.amount"] / pow(10, self.decimal_dict[src_chain][x["args.asset_s"].lower()])
            if x["args.asset_s"].lower() in self.decimal_dict[src_chain]
            else x["args.amount"],
            axis=1,
        )

        df = df[(df["value"] > 0) & (df["args.amount"] > 0)].copy()

        if not df.empty:
            df.loc[:, "amount_diff"] = df["args.amount"] - df["value"]
            df = df[df["amount_diff"] >= 0].copy()

            if not df.empty:
                df.loc[:, "threshold"] = df["amount_diff"] / df["args.amount"]

                while df[df["threshold"] <= threshold].empty and threshold < 1:
                    threshold *= 2

                df = df[df["threshold"] <= threshold].copy()

        return df

    def _match_timestamp(self, df: pd.DataFrame, threshold: float, key: str) -> pd.DataFrame:
        df.loc[:, "timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0)
        df.loc[:, "timeStamp"] = pd.to_numeric(df["timeStamp"], errors="coerce").fillna(0)

        df = df[df["timestamp"] < df["timeStamp"]].copy()

        if not df.empty:
            df.loc[:, "time_diff"] = df.apply(lambda x: x["timeStamp"] - x["timestamp"], axis=1)

            cur_threshold = threshold
            while df[df["time_diff"] <= cur_threshold].empty and cur_threshold < threshold * 2:
                cur_threshold += cur_threshold * 0.1

            scoped = df[df["time_diff"] <= cur_threshold].copy()
            idx = scoped.groupby(key, sort=False)["time_diff"].idxmin()
            df = scoped.loc[idx].reset_index(drop=True)

            df = df.drop(columns="time_diff")

        return df

    def _match_asset_type(self, df: pd.DataFrame) -> pd.DataFrame:
        if "args.asset_d" not in df.columns:
            return df
        ad = df["args.asset_d"].fillna("").astype(str).str.lower()
        ca = df["contractAddress"].fillna("").astype(str).str.lower()
        skip = ad.eq("") | ad.eq("nan")
        return df[skip | (ad == ca)].copy()

    def search_withdraw(self, fulloutput=False):
        res_df = pd.DataFrame()
        debug = {}
        for group in self.src_tx_group:
            src_chain, dst_chain = group[0]
            src_txs = group[1].reset_index(drop=True)

            tmp_df = src_txs.merge(self.dst_txs, how="cross")
            dst_txs_rows = 0 if self.dst_txs is None else len(self.dst_txs)

            def _count_by_txhash(df: pd.DataFrame) -> dict:
                if df is None or df.empty or "txhash" not in df.columns:
                    return {}
                return df.groupby("txhash").size().to_dict()

            merge_counts = _count_by_txhash(tmp_df)
            if tmp_df.empty:
                tmp_df.insert(loc=len(tmp_df.columns), column="hash", value="")
            else:
                tmp_df["contractAddress"] = tmp_df["contractAddress"].fillna("")

            if not tmp_df.empty:
                tmp_df = self._match_receiver(tmp_df)
            receiver_counts = _count_by_txhash(tmp_df)

            if not tmp_df.empty:
                tmp_df = self._match_tx_type(tmp_df)

            if not tmp_df.empty:
                tmp_df = self._match_asset_type(tmp_df)

            if not tmp_df.empty:
                time_threshold = float(os.environ.get("CONNECTOR_TIME_THRESHOLD", "1800"))
                tmp_df = self._match_timestamp(tmp_df, threshold=time_threshold, key="txhash")
            timestamp_counts = _count_by_txhash(tmp_df)

            if not tmp_df.empty:
                fee_threshold = float(os.environ.get("CONNECTOR_FEE_THRESHOLD", "0.03"))
                tmp_df = self._match_amount(tmp_df, src_chain, dst_chain, threshold=fee_threshold)
            amount_counts = _count_by_txhash(tmp_df)

            tmp_df = tmp_df.drop_duplicates(subset=["txhash"], keep="first")
            tmp_df = src_txs[["txhash"]].merge(tmp_df, left_on="txhash", right_on="txhash", how="left")
            res_df = pd.concat([res_df, tmp_df], ignore_index=True)

            if fulloutput and "txhash" in src_txs.columns:
                for h in src_txs["txhash"].astype(str).tolist():
                    debug[h] = {
                        "dst_txs_rows": dst_txs_rows,
                        "merge_rows": int(merge_counts.get(h, 0)),
                        "after_receiver_rows": int(receiver_counts.get(h, 0)),
                        "after_timestamp_rows": int(timestamp_counts.get(h, 0)),
                        "after_amount_rows": int(amount_counts.get(h, 0)),
                    }

        res_df = res_df[["args.srcChain", "txhash", "args.dstChain", "hash"]]
        res_df = res_df.rename(
            columns={
                "args.srcChain": "srcnet",
                "txhash": "srcTxHash",
                "args.dstChain": "dstnet",
                "hash": "dstTxHash",
            }
        )

        records = res_df.to_dict("records")
        if fulloutput:
            return records, debug
        return records
