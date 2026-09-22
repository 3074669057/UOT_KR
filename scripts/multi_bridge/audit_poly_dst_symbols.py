"""Query BSC-side token symbols for Poly dst tokens and identify canonical mapping."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from run_rc_uot_q_multi_bridge import load_bridge_cached, norm  # noqa: E402

cfg = json.loads((REPO / "config" / "local.json").read_text(encoding="utf-8"))
defaults = json.loads((REPO / "config" / "defaults.json").read_text(encoding="utf-8"))
nr = {**defaults.get("nodereal", {}), **cfg.get("nodereal", {})}
tmpl = str(nr.get("endpoint_template") or "https://bsc-mainnet.nodereal.io/v1/{api_key}")
urls = [tmpl.format(api_key=k) for k in (nr.get("api_keys") or [])]

SYMBOL_SELECTOR = "0x95d89b41"

def token_symbol(rpc_url: str, addr: str, timeout: int = 25) -> str:
    payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_call",
               "params": [{"to": addr, "data": SYMBOL_SELECTOR}, "latest"]}
    req = urllib.request.Request(rpc_url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode())
    if out.get("error"):
        return f"ERR"
    hexval = out.get("result") or "0x"
    if hexval in ("0x", "0x0"):
        return "<none>"
    data = bytes.fromhex(hexval[2:])
    if len(data) < 64:
        return "<short>"
    ln = int.from_bytes(data[32:64], "big")
    try:
        return data[64:64 + ln].decode("utf-8").rstrip("\x00")
    except Exception:
        return "<bin>"


cached = load_bridge_cached("Poly")
split = cached["split"]
dev = split[split["split"] == "development"]
all_c = pd.concat([cached["dev_candidates"], cached["test_candidates"]], ignore_index=True).drop_duplicates("candidate_tx_hash")
c_lookup = {norm(r.candidate_tx_hash): r for r in all_c.itertuples()}

# map src token -> dst tokens via dev truth
from collections import Counter, defaultdict
s2d: dict[str, Counter] = defaultdict(Counter)
for r in dev.itertuples():
    c = c_lookup.get(norm(r.dest_tx_hash))
    if c is None:
        continue
    s2d[norm(r.source_token_address)][norm(c.token_address)] += 1

dst_tokens = sorted({t for c in s2d.values() for t in c})
sym: dict[str, str] = {}
print(f"querying {len(dst_tokens)} Poly BSC-side tokens...")
for t in dst_tokens:
    for u in urls:
        try:
            sym[t] = token_symbol(u, t)
            break
        except Exception:
            sym[t] = "ERR"
    print(f"  {t} -> {sym[t]}")

# Poly src symbols from FirstPhrase
fp = pd.read_csv(REPO / "data" / "FirstPhrase" / "Poly_ETH.csv", dtype=str, keep_default_na=False)
fp["contractAddress"] = fp["contractAddress"].astype(str).str.lower()
src_syms: dict[str, str] = {}
for a in s2d:
    sub = fp[fp["contractAddress"] == a]
    if not sub.empty:
        syms = [s.split("_")[0] for s in sub["symbol"].unique() if s and s != "native_"]
        src_syms[a] = syms[0] if syms else ""
print("\nPoly src->dst mapping (top by count):")
rows = []
for a, c in sorted(s2d.items(), key=lambda kv: -sum(kv[1].values())):
    d, n = c.most_common(1)[0]
    rows.append((a, src_syms.get(a, "?"), d, sym.get(d, "?"), n, sum(c.values())))
for r in rows:
    print("  ", r)
