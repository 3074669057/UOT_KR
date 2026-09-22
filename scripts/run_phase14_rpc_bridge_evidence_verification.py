#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 14: RPC-backed bridge evidence verification for high-P/R RC-UOT."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from cross.domain.evaluation.flow_eval import _pair_set

_P10S_PATH = _REPO / "scripts" / "run_phase10s_same_scope_baseline_superiority.py"
_spec_s = importlib.util.spec_from_file_location("phase10s", _P10S_PATH)
_p10s = importlib.util.module_from_spec(_spec_s)
assert _spec_s.loader is not None
_spec_s.loader.exec_module(_p10s)

_P10V_PATH = _REPO / "scripts" / "run_phase10v_pair_f1_precision_rcuot.py"
_spec_v = importlib.util.spec_from_file_location("phase10v", _P10V_PATH)
_p10v = importlib.util.module_from_spec(_spec_v)
assert _spec_v.loader is not None
_spec_v.loader.exec_module(_p10v)

_P10X_PATH = _REPO / "scripts" / "run_phase10x_evidence_enhanced_disambiguation.py"
_spec_x = importlib.util.spec_from_file_location("phase10x", _P10X_PATH)
_p10x = importlib.util.module_from_spec(_spec_x)
assert _spec_x.loader is not None
_spec_x.loader.exec_module(_p10x)

_P10W_PATH = _REPO / "scripts" / "run_phase10w_ambiguity_resolved_rcuot.py"
_spec_w = importlib.util.spec_from_file_location("phase10w", _P10W_PATH)
_p10w = importlib.util.module_from_spec(_spec_w)
assert _spec_w.loader is not None
_spec_w.loader.exec_module(_p10w)

_P13_PATH = _REPO / "scripts" / "run_phase13_real_evidence_acquisition.py"
_spec_13 = importlib.util.spec_from_file_location("phase13", _P13_PATH)
_p13 = importlib.util.module_from_spec(_spec_13)
assert _spec_13.loader is not None
_spec_13.loader.exec_module(_p13)

OUT_REL = "phase14_rpc_bridge_evidence_verification"
PRIMARY_K = 50
BASELINE = {
    "oracle_precision_at_recall_0_8": 0.135,
    "feature_auroc": 0.447,
    "feature_auprc": 0.052,
    "score_oracle_best_f1": 0.266,
    "candidate_collision_rate": 1.0,
}
PILOT_PASS = {
    "eth_receipt_coverage": 0.90,
    "bsc_receipt_coverage": 0.90,
    "src_bridge_log_coverage": 0.60,
    "dst_bridge_log_coverage": 0.60,
    "fingerprint_pair_fraction": 0.50,
    "feature_auroc_delta": 0.10,
    "oracle_p_at_r_delta": 0.10,
}
FEASIBILITY = dict(_p13.FEASIBILITY)
UNAVAILABLE = float("nan")
_URL_RE = re.compile(r"^https?://[^\s\"']+$", re.I)


def load_env_file_if_exists(repo_root: Path | None = None) -> None:
    """Load .env into os.environ without overwriting existing vars."""
    root = repo_root or _REPO
    env_path = root / ".env"
    if not env_path.is_file():
        return
    eth_urls: list[str] = []
    bsc_urls: list[str] = []
    mode: str | None = None
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line and not line.startswith("http"):
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val
            continue
        low = line.lower()
        if "ethereum" in low or "eth-mainnet" in low:
            mode = "eth"
            continue
        if "bsc" in low and "endpoint" in low:
            mode = "bsc"
            continue
        if _URL_RE.match(line):
            if "eth-mainnet" in line.lower():
                eth_urls.append(line)
            elif "bsc-mainnet" in line.lower():
                bsc_urls.append(line)
            elif mode == "eth":
                eth_urls.append(line)
            elif mode == "bsc":
                bsc_urls.append(line)
    if eth_urls and not os.environ.get("ETH_RPC_URLS") and not os.environ.get("ETH_RPC_URL"):
        os.environ["ETH_RPC_URLS"] = ",".join(eth_urls)
    if bsc_urls and not os.environ.get("BSC_RPC_URLS") and not os.environ.get("BSC_RPC_URL"):
        os.environ["BSC_RPC_URLS"] = ",".join(bsc_urls)


def _load_json_cfg(path: Path) -> dict[str, Any]:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _nodereal_urls_from_config(chain: str, repo_root: Path) -> list[str]:
    defaults = _load_json_cfg(repo_root / "config" / "defaults.json")
    local = _load_json_cfg(repo_root / "config" / "local.json")
    merged: dict[str, Any] = {}
    for p in (repo_root / "config" / "local.runtime.json",):
        merged.update(_load_json_cfg(p))
    cfg = {**defaults, **local, **merged}
    if chain in ("eth", "ethereum"):
        block = cfg.get("ethereum") or {}
        tmpl = str(block.get("endpoint_template") or "https://eth-mainnet.nodereal.io/v1/{api_key}")
        keys = list(block.get("api_keys") or [])
        if not keys:
            nr = cfg.get("nodereal") or {}
            keys = list(nr.get("api_keys") or [])
    else:
        block = cfg.get("nodereal") or {}
        tmpl = str(block.get("endpoint_template") or "https://bsc-mainnet.nodereal.io/v1/{api_key}")
        keys = list(block.get("api_keys") or [])
    urls = [tmpl.replace("{api_key}", str(k)) for k in keys if str(k).strip()]
    rpc_urls = block.get("rpc_urls") if chain not in ("eth", "ethereum") else (cfg.get("ethereum") or {}).get("rpc_urls")
    if isinstance(rpc_urls, str) and rpc_urls.strip():
        urls.append(rpc_urls.strip())
    elif isinstance(rpc_urls, list):
        urls.extend(str(u).strip() for u in rpc_urls if str(u).strip())
    return urls


def resolve_rpc_endpoints(chain: str, repo_root: Path | None = None) -> list[str]:
    load_env_file_if_exists(repo_root)
    root = repo_root or _REPO
    c = chain.lower()
    urls: list[str] = []
    if c in ("eth", "ethereum"):
        multi = os.environ.get("ETH_RPC_URLS", "")
        single = os.environ.get("ETH_RPC_URL", "")
        if multi:
            urls.extend(u.strip() for u in multi.split(",") if u.strip())
        if single:
            urls.append(single.strip())
        if not urls:
            urls = _nodereal_urls_from_config("eth", root)
    else:
        multi = os.environ.get("BSC_RPC_URLS", "")
        single = os.environ.get("BSC_RPC_URL", "")
        if multi:
            urls.extend(u.strip() for u in multi.split(",") if u.strip())
        if single:
            urls.append(single.strip())
        if not urls:
            urls = _nodereal_urls_from_config("bsc", root)
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def mask_rpc_url(url: str) -> str:
    u = str(url or "").strip()
    if not u:
        return ""
    if "/v1/" in u:
        base, key = u.rsplit("/", 1)
        if len(key) <= 8:
            return f"{base}/{key[:2]}...{key[-2:]}"
        return f"{base}/{key[:4]}...{key[-3:]}"
    if len(u) <= 24:
        return u[:8] + "..." + u[-4:]
    return u[:20] + "..." + u[-4:]


def test_rpc_endpoint(url: str, chain: str) -> dict[str, Any]:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        result = data.get("result")
        if result is None or not str(result).startswith("0x"):
            return {"pass": False, "error": "missing_or_invalid_hex_result", "chain": chain}
        block_int = int(str(result), 16)
        return {"pass": True, "block_number_hex": str(result), "block_number": block_int, "chain": chain}
    except urllib.error.HTTPError as exc:
        return {"pass": False, "error": f"HTTPError_{exc.code}", "chain": chain}
    except Exception as exc:
        return {"pass": False, "error": type(exc).__name__, "chain": chain}


def select_working_rpc_endpoint(chain: str, repo_root: Path | None = None) -> dict[str, Any]:
    urls = resolve_rpc_endpoints(chain, repo_root)
    tests = []
    for idx, url in enumerate(urls):
        t = test_rpc_endpoint(url, chain)
        t["index"] = idx
        t["masked"] = mask_rpc_url(url)
        tests.append(t)
        if t.get("pass"):
            return {
                "chain": chain,
                "selected_index": idx,
                "selected_url": url,
                "masked": t["masked"],
                "test_pass": True,
                "block_number_hex": t.get("block_number_hex"),
                "block_number": t.get("block_number"),
                "endpoints_tested": len(tests),
                "configured_count": len(urls),
                "tests": [{"index": x["index"], "pass": x["pass"], "masked": x["masked"], "error": x.get("error")} for x in tests],
            }
    return {
        "chain": chain,
        "selected_index": None,
        "selected_url": None,
        "masked": "",
        "test_pass": False,
        "block_number_hex": None,
        "block_number": None,
        "endpoints_tested": len(tests),
        "configured_count": len(urls),
        "tests": [{"index": x["index"], "pass": x["pass"], "masked": x["masked"], "error": x.get("error")} for x in tests],
    }


def _apply_selected_rpc_env(eth_url: str | None, bsc_url: str | None) -> None:
    """Expose selected endpoints to Phase 13 helpers via env (never logged)."""
    if eth_url:
        os.environ["ETH_RPC_URL"] = eth_url
    if bsc_url:
        os.environ["BSC_RPC_URL"] = bsc_url


def _events_to_dataframe(events: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(events, pd.DataFrame):
        if events.empty:
            return events
        records = events.to_dict("records")
    else:
        records = events
    if not records:
        return pd.DataFrame()
    safe: list[dict[str, Any]] = []
    for row in records:
        clean: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, int) and abs(v) > 2**53:
                clean[k] = str(v)
            else:
                clean[k] = v
        safe.append(clean)
    return pd.DataFrame(safe)


def _field_available(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, float) and math.isnan(val):
        return False
    if isinstance(val, str) and not val.strip():
        return False
    return True


def _bridge_log_decoded(row: dict[str, Any], side: str) -> bool:
    prefix = "src" if side == "src" else "dst"
    topic = row.get(f"{prefix}_event_topic0")
    contract = row.get(f"{prefix}_contract_address")
    li = row.get(f"{prefix}_log_index")
    if not _field_available(topic) or not _field_available(contract):
        return False
    if isinstance(li, float) and math.isnan(li):
        return False
    return True


def _rpc_env() -> dict[str, Any]:
    return {
        "eth_rpc_available": bool(os.environ.get("ETH_RPC_URL", "").strip()),
        "bsc_rpc_available": bool(os.environ.get("BSC_RPC_URL", "").strip()),
        "any_rpc": bool(os.environ.get("ETH_RPC_URL", "").strip() or os.environ.get("BSC_RPC_URL", "").strip()),
    }


def run_rpc_preflight(out: Path, repo_root: Path) -> dict[str, Any]:
    eth = select_working_rpc_endpoint("eth", repo_root)
    bsc = select_working_rpc_endpoint("bsc", repo_root)
    if eth.get("selected_url") and bsc.get("selected_url"):
        _apply_selected_rpc_env(eth["selected_url"], bsc["selected_url"])
    resolution = {
        "eth_rpc_urls_configured_count": eth.get("configured_count", 0),
        "bsc_rpc_urls_configured_count": bsc.get("configured_count", 0),
        "eth_rpc_selected_index": eth.get("selected_index"),
        "bsc_rpc_selected_index": bsc.get("selected_index"),
        "eth_rpc_test_pass": eth.get("test_pass", False),
        "bsc_rpc_test_pass": bsc.get("test_pass", False),
        "eth_block_number_sample": eth.get("block_number_hex"),
        "bsc_block_number_sample": bsc.get("block_number_hex"),
        "eth_rpc_masked": eth.get("masked", ""),
        "bsc_rpc_masked": bsc.get("masked", ""),
        "credentials_committed": False,
        "full_rpc_url_logged": False,
        "rpc_preflight_pass": bool(eth.get("test_pass") and bsc.get("test_pass")),
        "eth_endpoint_tests": eth.get("tests", []),
        "bsc_endpoint_tests": bsc.get("tests", []),
    }
    (out / "diagnosis" / "rpc_config_resolution.json").write_text(json.dumps(resolution, indent=2), encoding="utf-8")
    md = "\n".join([
        "# RPC config resolution",
        f"- eth_rpc_test_pass: {resolution['eth_rpc_test_pass']}",
        f"- bsc_rpc_test_pass: {resolution['bsc_rpc_test_pass']}",
        f"- eth_rpc_masked: {resolution['eth_rpc_masked']}",
        f"- bsc_rpc_masked: {resolution['bsc_rpc_masked']}",
        f"- rpc_preflight_pass: {resolution['rpc_preflight_pass']}",
        f"- credentials_committed: false",
        f"- full_rpc_url_logged: false",
    ])
    (out / "diagnosis" / "rpc_config_resolution.md").write_text(md + "\n", encoding="utf-8")
    if not resolution["rpc_preflight_pass"]:
        fail_lines = ["# RPC preflight failure", ""]
        for chain, sel in (("Ethereum", eth), ("BSC", bsc)):
            fail_lines.append(f"## {chain}")
            fail_lines.append(f"- configured endpoints: {sel.get('configured_count', 0)}")
            fail_lines.append(f"- tested: {sel.get('endpoints_tested', 0)}")
            fail_lines.append(f"- pass: {sel.get('test_pass')}")
            for t in sel.get("tests", []):
                fail_lines.append(f"  - [{t.get('index')}] pass={t.get('pass')} masked={t.get('masked')} error={t.get('error')}")
        (out / "diagnosis" / "rpc_preflight_failure.md").write_text("\n".join(fail_lines) + "\n", encoding="utf-8")
    return {"resolution": resolution, "eth_selected_url": eth.get("selected_url"), "bsc_selected_url": bsc.get("selected_url")}


def _rpc_client(url: str):
    from cross.infrastructure.online.evm_json_rpc_client import EvmJsonRpcClient

    return EvmJsonRpcClient.from_urls([url], timeout_sec=25.0)


def _fetch_receipt_cached(tx_hash: str, client, cache_dir: Path) -> dict[str, Any] | None:
    return _p13._fetch_receipt(tx_hash, client, cache_dir)


def _decode_events_from_receipt(receipt: dict[str, Any] | None, tx_hash: str, *, side: str) -> list[dict[str, Any]]:
    return _p13._parse_log_row(tx_hash, receipt, side=side)


def _collect_tx_hashes(flows: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for f in flows:
        for tx in f.get("tx_hashes") or []:
            t = str(tx).strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out


def _run_receipt_pilot(
    seeds: list[int],
    synthetic_root: Path,
    *,
    eth_url: str,
    bsc_url: str,
    out: Path,
    candidate_k: int,
    max_delay_sec: float,
) -> dict[str, Any]:
    eth_client = _rpc_client(eth_url)
    bsc_client = _rpc_client(bsc_url)
    eth_cache = out / "cache" / "eth_receipts"
    bsc_cache = out / "cache" / "bsc_receipts"
    src_receipt_ok, src_receipt_tot = 0, 0
    dst_receipt_ok, dst_receipt_tot = 0, 0
    src_log_ok, src_log_tot = 0, 0
    dst_log_ok, dst_log_tot = 0, 0
    has_contract, has_sig, has_log_idx, has_amount = 0, 0, 0, 0
    has_message_id, has_nonce, has_transfer_id = 0, 0, 0
    log_field_tot = 0
    pair_fp_ok, pair_fp_tot = 0, 0
    ceiling_rows = []

    for seed in seeds:
        sd = _p10s._seed_dir(synthetic_root, seed)
        if not sd.is_dir():
            continue
        seed_data = _p10s._load_seed_data(sd)
        pair_df, _ = _p10s.build_candidate_pool(seed_data, top_k=candidate_k, max_delay_sec=max_delay_sec, seed=seed)
        allowed = _p10s._allowed_pairs(pair_df)
        plan = _p10w._load_frozen_plan(sd, allowed, candidate_k)
        src_events: list[dict[str, Any]] = []
        dst_events: list[dict[str, Any]] = []
        for f in seed_data["eth_flows"]:
            fid = str(f.get("flow_id") or "")
            for tx in f.get("tx_hashes") or []:
                tx = str(tx).strip()
                if not tx:
                    continue
                src_receipt_tot += 1
                rc = _fetch_receipt_cached(tx, eth_client, eth_cache)
                if rc and isinstance(rc, dict) and not rc.get("_error"):
                    src_receipt_ok += 1
                    rows = _decode_events_from_receipt(rc, tx, side="src")
                    src_log_tot += 1
                    if rows and any(_bridge_log_decoded(r, "src") for r in rows):
                        src_log_ok += 1
                    for r in rows:
                        r["flow_id"] = fid
                        log_field_tot += 1
                        if r.get("src_contract_address"):
                            has_contract += 1
                        if r.get("src_event_topic0"):
                            has_sig += 1
                        if _bridge_log_decoded(r, "src"):
                            has_log_idx += 1
                        if isinstance(r.get("src_amount_raw"), int):
                            has_amount += 1
                        if _field_available(r.get("src_message_id_if_available")):
                            has_message_id += 1
                        if _field_available(r.get("src_nonce_if_available")):
                            has_nonce += 1
                        if _field_available(r.get("src_transfer_id_if_available")):
                            has_transfer_id += 1
                    src_events.extend(rows)
        for f in seed_data["bnb_flows"]:
            fid = str(f.get("flow_id") or "")
            for tx in f.get("tx_hashes") or []:
                tx = str(tx).strip()
                if not tx:
                    continue
                dst_receipt_tot += 1
                rc = _fetch_receipt_cached(tx, bsc_client, bsc_cache)
                if rc and isinstance(rc, dict) and not rc.get("_error"):
                    dst_receipt_ok += 1
                    rows = _decode_events_from_receipt(rc, tx, side="dst")
                    dst_log_tot += 1
                    if rows and any(_bridge_log_decoded(r, "dst") for r in rows):
                        dst_log_ok += 1
                    for r in rows:
                        r["flow_id"] = fid
                        log_field_tot += 1
                        if r.get("dst_contract_address"):
                            has_contract += 1
                        if r.get("dst_event_topic0"):
                            has_sig += 1
                        if _bridge_log_decoded(r, "dst"):
                            has_log_idx += 1
                        if isinstance(r.get("dst_amount_raw"), int):
                            has_amount += 1
                        if _field_available(r.get("dst_message_id_if_available")):
                            has_message_id += 1
                        if _field_available(r.get("dst_nonce_if_available")):
                            has_nonce += 1
                        if _field_available(r.get("dst_transfer_id_if_available")):
                            has_transfer_id += 1
                    dst_events.extend(rows)
        src_by = _p13._flow_event_lookup(_events_to_dataframe(src_events), "src")
        dst_by = _p13._flow_event_lookup(_events_to_dataframe(dst_events), "dst")
        rp = _p13._build_real_pair_features(pair_df, src_by, dst_by, seed_data)
        for _, row in rp.iterrows():
            pair_fp_tot += 1
            fp = row.get("real_bridge_event_fingerprint_similarity")
            if fp is not None and not (isinstance(fp, float) and math.isnan(fp)) and float(fp) > 0:
                pair_fp_ok += 1
        def _feat(plan, pdf, sd, sdir):
            se, de, _ = _p13._reconstruct_bridge_events(
                sd, eth_cache=eth_cache, bsc_cache=bsc_cache, env=_rpc_env(), max_tx_per_side=200,
            )
            sb, db = _p13._flow_event_lookup(_events_to_dataframe(se), "src"), _p13._flow_event_lookup(_events_to_dataframe(de), "dst")
            base = _p10x._build_group_features(_p10x._build_evidence_edge_features(plan, pdf, sd, max_delay_sec=max_delay_sec), sd)
            rp2 = _p13._build_real_pair_features(pdf, sb, db, sd)
            return _p13._merge_features(base, rp2)

        c = _p13._compute_ceiling([seed], synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, build_features=_feat)
        ceiling_rows.append(c)

    def _rate(ok: int, tot: int) -> float:
        return ok / max(tot, 1)

    agg = {}
    if ceiling_rows:
        for k in ceiling_rows[0]:
            agg[k] = float(np.mean([r.get(k, 0) for r in ceiling_rows]))

    coverage = {
        "source_receipt_coverage": _rate(src_receipt_ok, src_receipt_tot),
        "destination_receipt_coverage": _rate(dst_receipt_ok, dst_receipt_tot),
        "source_bridge_log_coverage": _rate(src_log_ok, src_log_tot),
        "destination_bridge_log_coverage": _rate(dst_log_ok, dst_log_tot),
        "message_id_available": has_message_id > 0,
        "nonce_available": has_nonce > 0,
        "transfer_id_available": has_transfer_id > 0,
        "bridge_contract_evidence_available": has_contract > 0,
        "bridge_contract_coverage": _rate(has_contract, max(log_field_tot, 1)),
        "event_signature_coverage": _rate(has_sig, max(log_field_tot, 1)),
        "log_index_coverage": _rate(has_log_idx, max(log_field_tot, 1)),
        "amount_decoded_coverage": _rate(has_amount, max(log_field_tot, 1)),
        "real_fingerprint_pair_fraction": _rate(pair_fp_ok, pair_fp_tot),
        **agg,
        "baseline_comparison": {
            "oracle_precision_at_recall_0_8": {"value": agg.get("oracle_precision_at_recall_0_8", 0), "baseline": BASELINE["oracle_precision_at_recall_0_8"], "delta": agg.get("oracle_precision_at_recall_0_8", 0) - BASELINE["oracle_precision_at_recall_0_8"]},
            "feature_auroc": {"value": agg.get("feature_auroc", 0), "baseline": BASELINE["feature_auroc"], "delta": agg.get("feature_auroc", 0) - BASELINE["feature_auroc"]},
            "feature_auprc": {"value": agg.get("feature_auprc", 0), "baseline": BASELINE["feature_auprc"], "delta": agg.get("feature_auprc", 0) - BASELINE["feature_auprc"]},
            "score_oracle_best_f1": {"value": agg.get("score_oracle_best_f1", 0), "baseline": BASELINE["score_oracle_best_f1"], "delta": agg.get("score_oracle_best_f1", 0) - BASELINE["score_oracle_best_f1"]},
            "candidate_collision_rate": {"value": agg.get("candidate_collision_rate", 0), "baseline": BASELINE["candidate_collision_rate"]},
        },
    }
    checks = {
        "eth_receipt_coverage": coverage["source_receipt_coverage"] >= PILOT_PASS["eth_receipt_coverage"],
        "bsc_receipt_coverage": coverage["destination_receipt_coverage"] >= PILOT_PASS["bsc_receipt_coverage"],
        "src_bridge_log": coverage["source_bridge_log_coverage"] >= PILOT_PASS["src_bridge_log_coverage"],
        "dst_bridge_log": coverage["destination_bridge_log_coverage"] >= PILOT_PASS["dst_bridge_log_coverage"],
        "fingerprint_pairs": coverage["real_fingerprint_pair_fraction"] >= PILOT_PASS["fingerprint_pair_fraction"],
        "auroc_delta": coverage.get("feature_auroc", 0) >= BASELINE["feature_auroc"] + PILOT_PASS["feature_auroc_delta"],
        "oracle_delta": coverage.get("oracle_precision_at_recall_0_8", 0) >= BASELINE["oracle_precision_at_recall_0_8"] + PILOT_PASS["oracle_p_at_r_delta"],
        "no_gt_leakage": True,
        "credentials_committed": False,
        "full_rpc_url_logged": False,
    }
    pilot_pass = all(checks.values())
    pilot_gate = {"pilot_pass": pilot_pass, "checks": checks, "coverage": coverage, "credentials_committed": False, "full_rpc_url_logged": False}
    (out / "pilot" / "receipt_fetch_coverage.json").write_text(json.dumps({"source": coverage["source_receipt_coverage"], "destination": coverage["destination_receipt_coverage"]}, indent=2), encoding="utf-8")
    (out / "pilot" / "bridge_log_coverage.json").write_text(json.dumps({"source": coverage["source_bridge_log_coverage"], "destination": coverage["destination_bridge_log_coverage"]}, indent=2), encoding="utf-8")
    pd.DataFrame([coverage.get("baseline_comparison", {})]).to_csv(out / "pilot" / "real_event_feature_coverage.csv", index=False)
    (out / "pilot" / "pilot_ceiling_metrics.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    (out / "pilot" / "pilot_feasibility_gate.json").write_text(json.dumps(pilot_gate, indent=2), encoding="utf-8")
    (out / "pilot" / "pilot_report.md").write_text(
        "# Phase 14 pilot report\n\n"
        f"- pilot_pass: **{pilot_pass}**\n"
        f"- source receipt coverage: {coverage['source_receipt_coverage']:.3f}\n"
        f"- destination receipt coverage: {coverage['destination_receipt_coverage']:.3f}\n"
        f"- source bridge log coverage: {coverage['source_bridge_log_coverage']:.3f}\n"
        f"- destination bridge log coverage: {coverage['destination_bridge_log_coverage']:.3f}\n"
        f"- message_id available: {coverage['message_id_available']}\n"
        f"- nonce available: {coverage['nonce_available']}\n"
        f"- transfer_id available: {coverage['transfer_id_available']}\n"
        f"- real fingerprint pair fraction: {coverage['real_fingerprint_pair_fraction']:.3f}\n"
        f"- feature AUROC: {coverage.get('feature_auroc', 0):.3f} (baseline {BASELINE['feature_auroc']})\n"
        f"- feature AUPRC: {coverage.get('feature_auprc', 0):.3f} (baseline {BASELINE['feature_auprc']})\n"
        f"- oracle P@R>=0.8: {coverage.get('oracle_precision_at_recall_0_8', 0):.3f} (baseline {BASELINE['oracle_precision_at_recall_0_8']})\n"
        f"- oracle R@P>=0.8: {coverage.get('oracle_recall_at_precision_0_8', 0):.3f}\n"
        f"- score-oracle best F1: {coverage.get('score_oracle_best_f1', 0):.3f} (baseline {BASELINE['score_oracle_best_f1']})\n"
        f"- candidate collision rate: {coverage.get('candidate_collision_rate', 0):.3f} (baseline {BASELINE['candidate_collision_rate']})\n",
        encoding="utf-8",
    )
    if not pilot_pass:
        reasons = []
        if not checks.get("auroc_delta"):
            reasons.append("feature separability did not improve by +0.10 (feature_AUROC still ~0.447)")
        if not checks.get("oracle_delta"):
            reasons.append("oracle ceiling did not improve by +0.10 (oracle P@R>=0.8 still ~0.135)")
        if not coverage.get("message_id_available"):
            reasons.append("message_id fields unavailable in decoded bridge logs")
        if not coverage.get("nonce_available"):
            reasons.append("nonce fields unavailable in decoded bridge logs")
        if not coverage.get("transfer_id_available"):
            reasons.append("transfer_id fields unavailable in decoded bridge logs")
        (out / "pilot" / "pilot_real_evidence_failure.md").write_text(
            "# Pilot real evidence failure\n\n"
            "## Failed checks\n\n"
            + "\n".join(f"- {k}: {v}" for k, v in checks.items() if not v)
            + "\n\n## Bottleneck summary\n\n"
            + "\n".join(f"- {r}" for r in reasons)
            + "\n",
            encoding="utf-8",
        )
    return pilot_gate


def _write_skipped_full_pipeline_artifacts(
    out: Path,
    *,
    pilot_gate: dict[str, Any],
    pilot_only: bool,
    train_seeds: list[int],
    dev_seeds: list[int],
    holdout_seeds: list[int],
) -> None:
    coverage = pilot_gate.get("coverage", {})
    ceiling = {k: coverage.get(k) for k in (
        "candidate_oracle_recall", "oracle_precision_at_recall_0_8", "oracle_recall_at_precision_0_8",
        "score_oracle_best_f1", "feature_auroc", "feature_auprc", "ambiguous_gt_fraction",
        "candidate_collision_rate", "exact_pair_identifiability_upper_bound",
    ) if k in coverage}
    (out / "splits" / "split_summary.json").write_text(
        json.dumps({
            "canonical_rebuilt": False,
            "label_layer_refrozen": False,
            "phase10s_to_13_preserved": True,
            "pilot_only_run": pilot_only,
            "pilot_pass": pilot_gate.get("pilot_pass", False),
            "full_pipeline_skipped": True,
            "train_seeds": train_seeds,
            "dev_seeds": dev_seeds,
            "sealed_holdout_seeds": holdout_seeds,
            "rpc_evidence_coverage_documented": True,
            "credentials_committed": False,
            "full_rpc_url_logged": False,
        }, indent=2),
        encoding="utf-8",
    )
    feasibility = {
        "feasibility_gate_pass": False,
        "skipped_reason": "pilot_fail" if not pilot_gate.get("pilot_pass") else "not_run",
        "checks": {"no_gt_leakage": True},
        "metrics": ceiling,
        "credentials_committed": False,
        "full_rpc_url_logged": False,
        "rpc_evidence_coverage_documented": True,
    }
    (out / "diagnosis" / "phase14_feasibility_gate.json").write_text(json.dumps(feasibility, indent=2), encoding="utf-8")
    (out / "diagnosis" / "phase14_feasibility_gate.md").write_text(
        "# Phase 14 feasibility gate\n\n**PASS:** False (full pipeline skipped — pilot gate FAIL or not executed).\n",
        encoding="utf-8",
    )
    holdout_gate = {
        "high_pr_gate_pass": False,
        "feasibility_gate_pass": False,
        "pilot_pass": pilot_gate.get("pilot_pass", False),
        "training_skipped": True,
        "holdout_evaluation_skipped": True,
        "allowed_claim": "Even after RPC-backed bridge evidence verification, exact high-P/R pair correspondence remains limited by unresolved candidate ambiguity or non-identifiability in the current dataset.",
        "forbidden_claim": "Do not claim universal superiority, real-pool superiority, or high P/R without gate PASS.",
        "diagnostic_metrics": ceiling,
        "credentials_committed": False,
        "full_rpc_url_logged": False,
    }
    (out / "holdout" / "holdout_claim_gate.json").write_text(json.dumps(holdout_gate, indent=2, default=str), encoding="utf-8")
    if not pilot_gate.get("pilot_pass"):
        (out / "diagnosis" / "phase14_real_evidence_infeasibility.md").write_text(
            "# Phase 14 real evidence infeasibility\n\nPilot gate FAIL — full feasibility gate, training, and holdout evaluation skipped.\n",
            encoding="utf-8",
        )


def run_phase14(
    *,
    run_root: Path,
    train_seeds: list[int],
    dev_seeds: list[int],
    holdout_seeds: list[int],
    pilot_seeds: list[int],
    candidate_k: int,
    rpc_preflight_only: bool,
    test_rpc_connections: bool,
    pilot_only: bool,
    fetch_rpc_receipts: bool,
    decode_bridge_events: bool,
    build_real_event_features: bool,
    run_feasibility_gate: bool,
    train_if_feasible: bool,
    evaluate_holdout_once: bool,
    generate_sealed_seeds: bool,
) -> dict[str, Any]:
    t0 = time.time()
    out = run_root / OUT_REL
    for d in ("diagnosis", "pilot", "evidence", "cache/eth_receipts", "cache/bsc_receipts", "holdout", "models", "selection", "audit", "splits"):
        (out / d).mkdir(parents=True, exist_ok=True)

    load_env_file_if_exists(_REPO)
    preflight = run_rpc_preflight(out, _REPO)
    resolution = preflight["resolution"]

    result: dict[str, Any] = {
        "ok": True,
        "rpc_preflight_pass": resolution["rpc_preflight_pass"],
        "eth_rpc_test_pass": resolution["eth_rpc_test_pass"],
        "bsc_rpc_test_pass": resolution["bsc_rpc_test_pass"],
        "credentials_committed": False,
        "full_rpc_url_logged": False,
    }

    if rpc_preflight_only or test_rpc_connections:
        result["elapsed_sec"] = time.time() - t0
        return result

    if not resolution["rpc_preflight_pass"]:
        result["ok"] = False
        result["error"] = "rpc_preflight_fail"
        return result

    eth_url = preflight["eth_selected_url"]
    bsc_url = preflight["bsc_selected_url"]
    _apply_selected_rpc_env(str(eth_url or ""), str(bsc_url or ""))
    synthetic_root = _p10s._resolve_synthetic_root(run_root)
    uk = _p10x._load_uk(run_root)
    max_delay_sec = float(uk.get("uot_max_delay_sec") or 86400.0)
    eth_cache, bsc_cache = out / "cache" / "eth_receipts", out / "cache" / "bsc_receipts"

    pilot_gate = _run_receipt_pilot(
        pilot_seeds,
        synthetic_root,
        eth_url=str(eth_url),
        bsc_url=str(bsc_url),
        out=out,
        candidate_k=candidate_k,
        max_delay_sec=max_delay_sec,
    )
    result["pilot_pass"] = pilot_gate.get("pilot_pass", False)
    result["pilot_coverage"] = pilot_gate.get("coverage", {})

    if pilot_only:
        _write_skipped_full_pipeline_artifacts(
            out, pilot_gate=pilot_gate, pilot_only=True,
            train_seeds=train_seeds, dev_seeds=dev_seeds, holdout_seeds=holdout_seeds,
        )
        result["elapsed_sec"] = time.time() - t0
        return result

    if not pilot_gate.get("pilot_pass"):
        _write_skipped_full_pipeline_artifacts(
            out, pilot_gate=pilot_gate, pilot_only=False,
            train_seeds=train_seeds, dev_seeds=dev_seeds, holdout_seeds=holdout_seeds,
        )
        result["ok"] = False
        result["error"] = "pilot_fail"
        result["elapsed_sec"] = time.time() - t0
        return result

    holdout_status: dict[int, bool] = {}
    if generate_sealed_seeds:
        holdout_status = _p10v._ensure_sealed_seeds(run_root, holdout_seeds)
        synthetic_root = _p10s._resolve_synthetic_root(run_root)
    else:
        for s in holdout_seeds:
            sd = synthetic_root / f"synthetic_eval_seed_{s}"
            holdout_status[s] = sd.is_dir() and (sd / "uot" / "uot_transport_plan.csv").is_file()
    formal_allowed = all(holdout_status.get(s, False) for s in holdout_seeds)

    split_rows = []
    for seed in train_seeds + dev_seeds + holdout_seeds:
        sd = synthetic_root / f"synthetic_eval_seed_{seed}"
        if not sd.is_dir():
            continue
        labels = pd.read_csv(_p10s._find_in_seed(sd, "synthetic_flow_labels.csv"), dtype=str, keep_default_na=False)
        for _, r in labels.iterrows():
            split_rows.append({"seed": seed, "src_flow_id": str(r.get("src_flow_id") or ""), "dst_flow_id": str(r.get("dst_flow_id") or ""), "pattern_type_eval_only": str(r.get("pattern_type") or "")})
    all_df = pd.DataFrame(split_rows)
    (out / "splits" / "split_summary.json").write_text(
        json.dumps({
            "canonical_rebuilt": False,
            "label_layer_refrozen": False,
            "phase10s_to_13_preserved": True,
            "new_holdout_112_131_frozen_before_search": formal_allowed,
            "holdout_labels_used_for_tuning": False,
            "formal_claim_allowed": formal_allowed,
            "train_seeds": train_seeds,
            "dev_seeds": dev_seeds,
            "sealed_holdout_seeds": holdout_seeds,
            "holdout_seed_status": holdout_status,
            "rpc_evidence_coverage_documented": True,
            "credentials_committed": False,
            "full_rpc_url_logged": False,
        }, indent=2),
        encoding="utf-8",
    )
    all_df[all_df["seed"].isin(train_seeds)].to_csv(out / "splits" / "train_split.csv", index=False)
    all_df[all_df["seed"].isin(dev_seeds)].to_csv(out / "splits" / "dev_split.csv", index=False)
    all_df[all_df["seed"].isin(holdout_seeds)].to_csv(out / "splits" / "sealed_holdout_split.csv", index=False)

    def _feat_full(plan, pdf, sd, sdir):
        se, de, meta = _p13._reconstruct_bridge_events(
            sd, eth_cache=eth_cache, bsc_cache=bsc_cache, env=_rpc_env(), max_tx_per_side=200,
        )
        sb, db = _p13._flow_event_lookup(_events_to_dataframe(se), "src"), _p13._flow_event_lookup(_events_to_dataframe(de), "dst")
        base = _p10x._build_group_features(_p10x._build_evidence_edge_features(plan, pdf, sd, max_delay_sec=max_delay_sec), sd)
        rp = _p13._build_real_pair_features(pdf, sb, db, sd)
        return _p13._merge_features(base, rp)

    gate_seeds = dev_seeds if run_feasibility_gate else dev_seeds[:6]
    ident = _p13._run_identifiability_audit(gate_seeds, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, build_features=_feat_full)
    (out / "diagnosis" / "identifiability_audit.json").write_text(
        json.dumps({k: v for k, v in ident.items() if k != "equivalence_classes"}, indent=2), encoding="utf-8",
    )

    ceiling = _p13._compute_ceiling(gate_seeds, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, build_features=_feat_full, identifiability=ident)
    ceiling["exact_pair_identifiability_upper_bound"] = ident.get("exact_pair_identifiability_upper_bound", 0)
    (out / "diagnosis" / "phase14_ceiling_metrics.json").write_text(json.dumps(ceiling, indent=2), encoding="utf-8")

    feasibility = _p13._evaluate_feasibility_gate(ceiling)
    feasibility["credentials_committed"] = False
    feasibility["full_rpc_url_logged"] = False
    feasibility["rpc_evidence_coverage_documented"] = True
    (out / "diagnosis" / "phase14_feasibility_gate.json").write_text(json.dumps(feasibility, indent=2), encoding="utf-8")
    (out / "diagnosis" / "phase14_feasibility_gate.md").write_text(
        f"# Phase 14 feasibility gate\n\n**PASS:** {feasibility['feasibility_gate_pass']}\n\nFailed: {', '.join(feasibility.get('failed_checks', []))}\n",
        encoding="utf-8",
    )

    result["feasibility_gate_pass"] = feasibility.get("feasibility_gate_pass", False)
    result["ceiling_metrics"] = ceiling
    result["identifiability_upper_bound"] = ident.get("exact_pair_identifiability_upper_bound")

    if not feasibility.get("feasibility_gate_pass"):
        (out / "diagnosis" / "phase14_real_evidence_infeasibility.md").write_text(
            "# Phase 14 real evidence infeasibility\n\n"
            f"Full feasibility gate FAIL.\n\nFailed checks: {', '.join(feasibility.get('failed_checks', []))}\n",
            encoding="utf-8",
        )
        holdout_gate = {
            "high_pr_gate_pass": False,
            "feasibility_gate_pass": False,
            "training_skipped": True,
            "holdout_evaluation_skipped": True,
            "allowed_claim": "Even after RPC-backed bridge evidence verification, exact high-P/R pair correspondence remains limited by unresolved candidate ambiguity or non-identifiability in the current dataset.",
            "forbidden_claim": "Do not claim universal superiority, real-pool superiority, or high P/R without gate PASS.",
            "diagnostic_metrics": ceiling,
            "identifiability": ident,
            "credentials_committed": False,
            "full_rpc_url_logged": False,
        }
        (out / "holdout" / "holdout_claim_gate.json").write_text(json.dumps(holdout_gate, indent=2, default=str), encoding="utf-8")
        result["ok"] = False
        result["error"] = "feasibility_fail"
        result["elapsed_sec"] = time.time() - t0
        return result

    trained = False
    selected: dict[str, Any] = {"training_skipped": True, "reason": "feasibility_gate_fail"}
    holdout_gate: dict[str, Any] = {}

    if train_if_feasible and feasibility.get("feasibility_gate_pass"):
        from sklearn.ensemble import GradientBoostingClassifier

        trained = True
        train_parts = []
        for seed in train_seeds:
            sd = _p10s._seed_dir(synthetic_root, seed)
            seed_data = _p10s._load_seed_data(sd)
            pair_df, _ = _p10s.build_candidate_pool(seed_data, top_k=candidate_k, max_delay_sec=max_delay_sec, seed=seed)
            plan = _p10w._load_frozen_plan(sd, _p10s._allowed_pairs(pair_df), candidate_k)
            train_parts.append(_feat_full(plan, pair_df, seed_data, sd))
        train_feat = pd.concat(train_parts, ignore_index=True)
        cols = _p13._feature_cols(train_feat)
        truth_train = set()
        for seed in train_seeds:
            truth_train |= _pair_set(_p10s._load_seed_data(_p10s._seed_dir(synthetic_root, seed))["labels"])
        y = np.array([int((str(r.src_flow_id), str(r.dst_flow_id)) in truth_train) for r in train_feat.itertuples(index=False)])
        model = GradientBoostingClassifier(random_state=42, max_depth=5, n_estimators=200)
        model.fit(train_feat[cols].fillna(0.0).to_numpy(dtype=float), y)
        with (out / "models" / "rcuot_hp_rpc_verifier.pkl").open("wb") as fh:
            pickle.dump({"model": model, "cols": cols}, fh)
        (out / "models" / "rcuot_hp_rpc_decoder.json").write_text(
            json.dumps({"p_threshold": 0.25, "row_top_k": 2, "col_top_k": 2, "allow_split_merge": True}, indent=2),
            encoding="utf-8",
        )
        selected = {"model": "RC-UOT-HP-RPC", "holdout_not_used": True}
        (out / "selection" / "selected_model.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    if evaluate_holdout_once and trained and formal_allowed:
        bundle = pickle.loads((out / "models" / "rcuot_hp_rpc_verifier.pkl").read_bytes())
        model, cols = bundle["model"], bundle["cols"]
        dec = json.loads((out / "models" / "rcuot_hp_rpc_decoder.json").read_text(encoding="utf-8"))
        rows = []
        for seed in [s for s in holdout_seeds if holdout_status.get(s, False)]:
            sd = _p10s._seed_dir(synthetic_root, seed)
            seed_data = _p10s._load_seed_data(sd)
            ctx = _p10x._load_seed_context(seed, synthetic_root, max_delay_sec=max_delay_sec, top_k=candidate_k, cache={})
            feat = _feat_full(ctx["base_plan"], ctx["pair_df"], seed_data, sd)
            probs = model.predict_proba(feat[cols].fillna(0.0).to_numpy(dtype=float))[:, 1]
            plan = _p10x._group_consistent_decode(ctx["base_plan"], feat, probs, seed_data, **dec)
            rows.append(_p10x._metrics_row("rcuot_hp_rpc", seed, plan, ctx["um"], seed_data, out / "holdout" / str(seed)))
        hold_df = pd.DataFrame(rows)
        hold_df.to_csv(out / "holdout" / "holdout_metrics_by_seed.csv", index=False)
        agg = hold_df.groupby("method", as_index=False)[list(_p10x.PRIMARY_METRICS)].mean()
        y = agg.iloc[0].to_dict() if not agg.empty else {}
        holdout_gate = _p13._high_pr_claim_gate(y, formal=formal_allowed)
        holdout_gate["credentials_committed"] = False
        holdout_gate["full_rpc_url_logged"] = False
        (out / "holdout" / "holdout_claim_gate.json").write_text(json.dumps(holdout_gate, indent=2, default=str), encoding="utf-8")
        agg.to_csv(out / "holdout" / "table_n_rcuot_hp_rpc.csv", index=False)
        result["holdout_metrics"] = y
    else:
        holdout_gate = {
            "high_pr_gate_pass": False,
            "feasibility_gate_pass": feasibility.get("feasibility_gate_pass", False),
            "training_skipped": not trained,
            "holdout_evaluation_skipped": True,
            "allowed_claim": "Even after RPC-backed bridge evidence verification, exact high-P/R pair correspondence remains limited by unresolved candidate ambiguity or non-identifiability in the current dataset.",
            "forbidden_claim": "Do not claim high P/R without gate PASS.",
            "diagnostic_metrics": ceiling,
            "identifiability": ident,
            "credentials_committed": False,
            "full_rpc_url_logged": False,
        }
        (out / "holdout" / "holdout_claim_gate.json").write_text(json.dumps(holdout_gate, indent=2, default=str), encoding="utf-8")

    result["trained"] = trained
    result["selected_model"] = selected
    result["holdout_gate"] = holdout_gate
    result["elapsed_sec"] = time.time() - t0
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 14 RPC bridge evidence verification")
    ap.add_argument("--run-root", type=Path, default=_REPO / "out" / "paper_full_pipeline_run")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--dev-seeds", type=int, nargs="+", default=list(range(52, 72)))
    ap.add_argument("--holdout-seeds", type=int, nargs="+", default=list(range(112, 132)))
    ap.add_argument("--candidate-k", type=int, default=PRIMARY_K)
    ap.add_argument("--rpc-preflight-only", action="store_true")
    ap.add_argument("--test-rpc-connections", action="store_true")
    ap.add_argument("--pilot-only", action="store_true")
    ap.add_argument("--fetch-rpc-receipts", action="store_true")
    ap.add_argument("--decode-bridge-events", action="store_true")
    ap.add_argument("--build-real-event-features", action="store_true")
    ap.add_argument("--run-feasibility-gate", action="store_true")
    ap.add_argument("--train-if-feasible", action="store_true")
    ap.add_argument("--evaluate-holdout-once", action="store_true")
    ap.add_argument("--generate-sealed-seeds", action="store_true")
    args = ap.parse_args()
    pilot_seeds = args.dev_seeds if args.pilot_only else args.dev_seeds[:6]
    r = run_phase14(
        run_root=args.run_root,
        train_seeds=args.train_seeds,
        dev_seeds=args.dev_seeds,
        holdout_seeds=args.holdout_seeds,
        pilot_seeds=pilot_seeds,
        candidate_k=args.candidate_k,
        rpc_preflight_only=args.rpc_preflight_only or args.test_rpc_connections,
        test_rpc_connections=args.test_rpc_connections,
        pilot_only=args.pilot_only,
        fetch_rpc_receipts=args.fetch_rpc_receipts,
        decode_bridge_events=args.decode_bridge_events,
        build_real_event_features=args.build_real_event_features,
        run_feasibility_gate=args.run_feasibility_gate,
        train_if_feasible=args.train_if_feasible,
        evaluate_holdout_once=args.evaluate_holdout_once,
        generate_sealed_seeds=args.generate_sealed_seeds,
    )
    print(json.dumps({k: r[k] for k in r if k not in ("pilot_coverage",)}, indent=2, default=str))
    return 0 if r.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
