"""Audit: variation sources for evidence quality within priced dev subsets (FirstPhrase corroboration, receiver presence, tx multiplicity)."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from run_rc_uot_q_multi_bridge import load_bridge_cached, norm  # noqa: E402

prices = json.loads((REPO / "data" / "Token" / "token_prices_usd.json").read_text(encoding="utf-8"))
prices = {str(k).lower(): v["usd"] for k, v in prices.items()}

# canonical mappings established in audit
CANON = {
    # Multi ETH side
    "0x0615dbba33fe61a31c7ed131bda6655ed76748b1": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # anyETH -> WETH
    "0x22648c12acd87912ea1710357b1302c6a4154ebc": "0xdac17f958d2ee523a2206206994597c13d831ec7",  # anyUSDT -> USDT
    # Multi BSC side
    "0xdebb1d6a2196f2335ad51fbde7ca587205889360": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # anyETH(BSC) -> ETH
    "0xedf0c420bc3b92b961c6ec411cc810ca81f5f21a": "0xdac17f958d2ee523a2206206994597c13d831ec7",  # anyUSDT(BSC) -> USDT
    "0x5fe80d2cd054645b9419657d3d10d26391780a7b": "0x4e352cf164e64adcbad318c3a1e222e9eba4ce42",  # MCB(BSC) -> MCB
    "0xb12c13e66ade1f72f71834f2fc5082db8c091358": "0x6b175474e89094c44da98b954eedeac495271d0f",  # DAI(BSC) -> DAI
    # Celer dst WETH-BSC -> WETH
    "0x2170ed0880ac9a755fd29b2688956bd959f933f8": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
}

def price_of(addr: str) -> float | None:
    a = norm(addr)
    if a in prices:
        return prices[a]
    if a in CANON:
        return prices.get(CANON[a])
    return None


for b in ("Celer", "Multi", "Poly"):
    cached = load_bridge_cached(b)
    split = cached["split"]
    dev = split[split["split"] == "development"]
    all_c = pd.concat([cached["dev_candidates"], cached["test_candidates"]], ignore_index=True).drop_duplicates("candidate_tx_hash")
    c_lookup = {norm(r.candidate_tx_hash): r for r in all_c.itertuples()}

    fp = pd.read_csv(REPO / "data" / "FirstPhrase" / f"{b}_ETH.csv", dtype=str, keep_default_na=False)
    fp["hash"] = fp["hash"].astype(str).str.lower()
    fp_hashes = set(fp["hash"].unique())
    fp_txcount = Counter(fp["hash"])

    sample = json.loads((REPO / "data" / "Validation" / "ETH-BNB" / b / "sample.json").read_text(encoding="utf-8"))
    recv_by_tx = {norm(it.get("txhash")): str((it.get("args") or {}).get("receiver") or "") for it in sample}
    send_by_tx = {norm(it.get("txhash")): str((it.get("args") or {}).get("sender") or "") for it in sample}

    priced = 0
    unpriced = 0
    fp_hit = 0
    fp_miss = 0
    recv_missing_dst = 0
    recv_ok_dst = 0
    fp_multi = 0
    self_bridge = 0
    delay_gt_6h = 0
    delays: list[float] = []
    n_rows = 0
    for r in dev.itertuples():
        n_rows += 1
        sh = norm(r.source_tx_hash)
        stok = norm(r.source_token_address)
        c = c_lookup.get(norm(r.dest_tx_hash))
        if c is None:
            continue
        dtok = norm(c.token_address)
        if price_of(stok) is None or price_of(dtok) is None:
            unpriced += 1
            continue
        priced += 1
        if sh in fp_hashes:
            fp_hit += 1
        else:
            fp_miss += 1
        if fp_txcount.get(sh, 0) > 1:
            fp_multi += 1
        if str(c.receiver).strip():
            recv_ok_dst += 1
        else:
            recv_missing_dst += 1
        send = send_by_tx.get(sh, "")
        recv = recv_by_tx.get(sh, "")
        if send and recv and send.lower() == recv.lower():
            self_bridge += 1
        delay = float(c.candidate_timestamp) - float(r.source_timestamp)
        delays.append(delay)
        if delay > 21600:
            delay_gt_6h += 1
    print(f"{b}: dev_rows={n_rows} priced={priced} unpriced={unpriced}")
    if priced:
        print(f"   src FirstPhrase hit={fp_hit} miss={fp_miss} multi-tx={fp_multi}")
        print(f"   dst receiver ok={recv_ok_dst} missing={recv_missing_dst}")
        print(f"   self-bridge (src sender==receiver)={self_bridge}")
        print(f"   delay>6h={delay_gt_6h} delay q=[{pd.Series(delays).quantile(0.1):.0f},{pd.Series(delays).quantile(0.5):.0f},{pd.Series(delays).quantile(0.9):.0f}]")
    print()
