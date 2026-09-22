"""Aggregate (a,b) edges into relationship buckets and ambiguous listings."""
from __future__ import annotations

import pandas as pd


def aggregate_relations(pairs_df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    amb_cols = ["address_a", "address_b", "srcTxHash", "dstTxHash", "kind"]
    empty_ambiguous = pd.DataFrame(columns=amb_cols)
    empty_edges = pd.DataFrame(columns=["address_a", "address_b", "tx_count"])
    zero_summary = {
        "edge_count": 0,
        "distinct_a": 0,
        "distinct_b": 0,
        "one_to_one_events": 0,
        "unique_one_to_one_pairs": 0,
        "edges_flagged_one_to_many": 0,
        "edges_flagged_many_to_one": 0,
        "addresses_with_one_to_many": 0,
        "addresses_with_many_to_one": 0,
    }
    if pairs_df.empty:
        return zero_summary, empty_ambiguous, empty_edges
    df = pairs_df[pairs_df["complete_ab"] == True].copy()  # noqa: E712
    df = df[df["address_a"].astype(str).str.len() > 0]
    df = df[df["address_b"].astype(str).str.len() > 0]
    if df.empty:
        return zero_summary, empty_ambiguous, empty_edges
    edge_count = len(df)
    distinct_a = df["address_a"].nunique()
    distinct_b = df["address_b"].nunique()
    a_ndistinct_b = df.groupby("address_a")["address_b"].nunique()
    b_ndistinct_a = df.groupby("address_b")["address_a"].nunique()

    def classify(row):
        a, b = row["address_a"], row["address_b"]
        o2m = bool(a_ndistinct_b.loc[a] > 1)
        m2o = bool(b_ndistinct_a.loc[b] > 1)
        kind = "both" if o2m and m2o else ("one_to_many" if o2m else ("many_to_one" if m2o else "one_to_one"))
        return pd.Series({"kind": kind, "one_to_many_a": o2m, "many_to_one_b": m2o})

    cls = df.apply(classify, axis=1)
    df = pd.concat([df.reset_index(drop=True), cls], axis=1)
    one_to_one_events = int((df["kind"] == "one_to_one").sum())
    valid_a = set(a_ndistinct_b[a_ndistinct_b == 1].index)
    valid_b = set(b_ndistinct_a[b_ndistinct_a == 1].index)
    distinct_pairs = df.groupby(["address_a", "address_b"], as_index=False).size()
    unique_one_to_one_pairs = int(
        distinct_pairs[
            distinct_pairs["address_a"].isin(valid_a) & distinct_pairs["address_b"].isin(valid_b)
        ].shape[0]
    )
    ambiguous = df[df["kind"] != "one_to_one"].copy()
    summary = {
        "edge_count": edge_count,
        "distinct_a": int(distinct_a),
        "distinct_b": int(distinct_b),
        "one_to_one_events": one_to_one_events,
        "unique_one_to_one_pairs": unique_one_to_one_pairs,
        "edges_flagged_one_to_many": int(df["one_to_many_a"].sum()),
        "edges_flagged_many_to_one": int(df["many_to_one_b"].sum()),
        "addresses_with_one_to_many": int((a_ndistinct_b > 1).sum()),
        "addresses_with_many_to_one": int((b_ndistinct_a > 1).sum()),
    }
    ambiguous_out = ambiguous[amb_cols] if not ambiguous.empty else empty_ambiguous
    edges_pair = df.groupby(["address_a", "address_b"], as_index=False).agg(tx_count=("srcTxHash", "count"))
    return summary, ambiguous_out, edges_pair
