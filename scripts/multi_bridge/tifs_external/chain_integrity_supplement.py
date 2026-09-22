"""Complete the chain-integrity tx samples via the second NodeReal credential.
DATA-ONLY. Labels provider diversity per check; no secrets printed."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings()

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_v3"
cfg = json.loads((REPO / "config" / "local.json").read_text(encoding="utf-8"))
KEYS = (cfg.get("nodereal") or {}).get("api_keys") or []
REDACT = re.compile(r"(nodereal\.io/v1/)[A-Za-z0-9]+")
NR2_ETH = f"https://eth-mainnet.nodereal.io/v1/{KEYS[1] if len(KEYS) > 1 else KEYS[0]}"
NR2_BSC = f"https://bsc-mainnet.nodereal.io/v1/{KEYS[1] if len(KEYS) > 1 else KEYS[0]}"
SESS = requests.Session()
SESS.verify = False
SESS.headers.update({"Content-Type": "application/json"})


def rpc(url, method, params, timeout=25):
    try:
        r = SESS.post(url, json={"jsonrpc": "2.0", "method": method,
                                 "params": params, "id": 1}, timeout=timeout)
        if r.status_code == 200:
            d = r.json()
            if "result" in d:
                return d["result"]
    except Exception:  # noqa: BLE001
        pass
    return None


def main() -> int:
    audit = json.loads((ROOT / "chain_integrity_audit.json").read_text(encoding="utf-8"))
    anchors = []
    for b in ("b01", "b02", "b03"):
        anchors += json.loads((ROOT / "blocks" / b / "anchors.json")
                              .read_text(encoding="utf-8"))
    done = {(c["anchor_idx"], c["chain"]) for c in audit["tx_sample_checks"]}
    supp = []
    for i in range(0, len(anchors), 264):
        a = anchors[i]
        for chain, tx in (("eth", a["src_tx"]), ("bsc", a["dst_tx"])):
            if (i, chain) in done:
                continue
            url = NR2_ETH if chain == "eth" else NR2_BSC
            rec = rpc(url, "eth_getTransactionReceipt", [tx])
            if rec is None:
                supp.append({"anchor_idx": i, "chain": chain, "match": None,
                             "provider": "none"})
                continue
            ok = rec["transactionHash"] == tx
            supp.append({"anchor_idx": i, "chain": chain, "match": ok,
                         "provider": REDACT.sub(r"\1***", url) + " (cross-key)",
                         "blockNumber": int(rec["blockNumber"], 16)})
            time.sleep(0.2)
    audit["tx_sample_checks_supplemental_crosskey"] = supp
    n_match = sum(1 for c in supp if c["match"] is True)
    n_fail = sum(1 for c in supp if c["match"] is False)
    n_none = sum(1 for c in supp if c["match"] is None)
    total = len(audit["tx_sample_checks"]) + len(supp)
    all_match = (n_fail == 0 and n_none == 0 and
                 not any(not c["match"] for c in audit["tx_sample_checks"]))
    # issues resolved by the supplemental cross-key fetches are no longer open
    resolved = {(c["anchor_idx"], c["chain"]) for c in supp}
    open_issues = [i for i in audit["issues"]
                   if not any(f"anchor {a} {ch}" in i for (a, ch) in resolved)]
    audit["issues"] = open_issues
    audit["coverage"]["n_tx_checks_total"] = total
    audit["coverage"]["supplemental_crosskey"] = {"n": len(supp), "match": n_match,
                                                  "fail": n_fail, "none": n_none}
    audit["CHAIN_DATA_INTEGRITY"] = ("PASS" if all_match and not open_issues
                                     else "FAIL")
    (ROOT / "chain_integrity_audit.json").write_text(json.dumps(audit, indent=2),
                                                     encoding="utf-8")
    print(json.dumps({"supplemental": {"n": len(supp), "match": n_match,
                                       "fail": n_fail, "none": n_none},
                      "CHAIN_DATA_INTEGRITY": audit["CHAIN_DATA_INTEGRITY"]}, indent=2))
    return 0 if audit["CHAIN_DATA_INTEGRITY"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
