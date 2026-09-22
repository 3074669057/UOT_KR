"""Final RC-UOT-Q development protocol with fail-closed provenance boundaries."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from cross.domain.uot.uot_solver_numpy import uot_sinkhorn

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "out" / "ec_uot_q_final"
V2_OUT = REPO / "out" / "ec_uot_q_final_v2"
V3_OUT = REPO / "out" / "ec_uot_q_final_v3"
SPLIT_MANIFEST = REPO / "out" / "bsc_open_independent_v1" / "stage2_split" / "stage2_split_manifest.json"
ASSIGNMENTS = REPO / "out" / "bsc_open_independent_v1" / "stage2_split" / "chronological_split_assignments.csv"
RAW_RELAY = REPO / "out" / "bsc_open_independent_v1" / "stage5_5_candidate_completion" / "raw_relay_logs_dev_v2.json"
ETH_PUBLIC = REPO / "data" / "in" / "Celer_ETH_cun.csv"

FORBIDDEN_FIELDS = {
    "label_dsttxhash", "is_truth", "message_key", "message_id", "nonce",
    "transfer_id", "src_transfer_id", "gt_receiver", "gt_amount", "gt_asset",
    "true_dst", "expected_dst", "oracle_", "label_candidate", "ground_truth_pair",
    "paired_tx", "matched_tx", "route_id",
    "transfer_key", "message_key", "bridge_key",
}
BRIDGE_FIELDS = {"route_id", "transfer_key", "bridge_key", "message_key", "message_id", "transfer_id", "src_transfer_id", "nonce"}
SOURCE_ALLOWLIST = {"source_tx_hash", "source_timestamp", "source_amount_raw", "source_token_address", "source_route_type"}
TARGET_ALLOWLIST = {"candidate_tx_hash", "candidate_timestamp", "amount_raw", "token_address", "contract_address", "topic0", "route_type", "bridge_contract_hit"}


# --- Public bridge/token registries (verifiable on-chain, no label dependency) ---

KNOWN_ETH_TOKEN_ADDRESSES = {
    "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
    "0x4fabb145d64652a948d72533023f6e7a623c7c53",  # BUSD
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
    "0x853d955acef822db058eb8505911ed77f175b99e",  # FRAX
}

CELER_ETH_BRIDGE_ADDRESSES = {
    "0xcafe001067cdef266afb7eb5a286dcfd277f3de5",
    "0x45c2f8c9b4c0bdc76200448cc26c48ab6ffef83f",
    "0x43dfc4159d86f3a37a5a4b3d4580b888ad7d4ddd",
    "0xca37530e7c5968627be470081d1c993eb1deaf90",
    "0x4e352cf164e64adcbad318c3a1e222e9eba4ce42",
}

RELAY_CONTRACT_BSC = "0xdd90e5e87a2081dcf0391920868ebc2ffb81a1af"
RELAY_TOPIC0 = "0x79fa08de5149d912dce8e5e8da7a7c17ccdf23dd5d3bfe196802e6eb86347c7c"

_PUBLIC_REGISTRY_SHA256 = None


def classify_route_type_source(contract_address: str) -> str:
    addr = _norm_hash(contract_address)
    if addr in KNOWN_ETH_TOKEN_ADDRESSES:
        return "token"
    if addr in CELER_ETH_BRIDGE_ADDRESSES:
        return "bridge"
    return "unknown"


def classify_route_type_target(contract_address: str, topic0: str) -> str:
    ca = _norm_hash(contract_address)
    t0 = _norm_hash(topic0)
    if ca == RELAY_CONTRACT_BSC and t0 == RELAY_TOPIC0:
        return "relay"
    return "unknown"


def pub_registry_sha256() -> str:
    global _PUBLIC_REGISTRY_SHA256
    if _PUBLIC_REGISTRY_SHA256 is None:
        import json as _json, hashlib as _hashlib
        payload = _json.dumps({
            "known_eth_token_addresses": sorted(KNOWN_ETH_TOKEN_ADDRESSES),
            "celer_eth_bridge_addresses": sorted(CELER_ETH_BRIDGE_ADDRESSES),
            "relay_contract_bsc": RELAY_CONTRACT_BSC,
            "relay_topic0": RELAY_TOPIC0,
        }, sort_keys=True)
        _PUBLIC_REGISTRY_SHA256 = _hashlib.sha256(payload.encode()).hexdigest()
    return _PUBLIC_REGISTRY_SHA256

class ProtocolError(RuntimeError):
    pass
class ForbiddenInferenceField(ProtocolError):
    pass
class AccessViolation(ProtocolError):
    pass
class HeldoutGateError(ProtocolError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def _norm_hash(value: Any) -> str:
    return str(value or "").strip().lower()


def _hex_int(value: Any) -> int:
    s = str(value or "0").strip()
    return int(s, 16) if s.lower().startswith("0x") else int(float(s or 0))


def _addr_word(data: str, index: int) -> str:
    raw = data[2:] if data.lower().startswith("0x") else data
    start = index * 64
    return "0x" + raw[start + 24:start + 64].lower()


def _uint_word(data: str, index: int) -> int:
    raw = data[2:] if data.lower().startswith("0x") else data
    start = index * 64
    return int(raw[start:start + 64], 16)


def decode_public_relay_log(log: dict[str, Any]) -> dict[str, Any]:
    """Decode public Relay data while intentionally discarding word 0 and word 5."""
    data = str(log.get("data") or "")
    raw = data[2:] if data.lower().startswith("0x") else data
    if len(raw) < 6 * 64:
        raise ProtocolError("Relay data is shorter than six ABI words")
    ca = _norm_hash(log.get("address"))
    t0 = _norm_hash((ast.literal_eval(log["topics"]) if isinstance(log.get("topics"), str) else (log.get("topics") or [""]))[0])
    return {
        "candidate_tx_hash": _norm_hash(log.get("transactionHash")),
        "candidate_timestamp": _hex_int(log.get("blockTimestamp")),
        "amount_raw": _uint_word(data, 4),
        "receiver": _addr_word(data, 2),
        "token_address": _addr_word(data, 3),
        "block_number": _hex_int(log.get("blockNumber")),
        "transaction_index": _hex_int(log.get("transactionIndex")),
        "log_index": _hex_int(log.get("logIndex")),
        "contract_address": ca,
        "topic0": t0,
        "route_type": classify_route_type_target(ca, t0),
        "bridge_contract_hit": True,
    }


def _iter_json_array(path: Path):
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        started = False
        eof = False
        while True:
            if not eof and len(buffer) < 1024 * 1024:
                chunk = handle.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            buffer = buffer.lstrip()
            if not started:
                if not buffer.startswith("["):
                    raise ProtocolError("raw Relay log file must be a JSON array")
                buffer = buffer[1:]
                started = True
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise ProtocolError("invalid raw Relay JSON")
                continue
            yield value
            buffer = buffer[end:]

def build_public_relay_candidates(raw_path: Path, *, max_timestamp_exclusive: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build candidate attributes only from raw public Relay logs."""
    raw_path = raw_path.resolve()
    rows = []
    for item in _iter_json_array(raw_path):
        decoded = decode_public_relay_log(item)
        if max_timestamp_exclusive is not None and decoded["candidate_timestamp"] >= max_timestamp_exclusive:
            break
        rows.append(decoded)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ProtocolError("No public Relay candidates decoded")
    frame = frame.drop_duplicates("candidate_tx_hash", keep="first").reset_index(drop=True)
    frame["quotient_group_id"] = [
        hashlib.sha256(f"{t}|{int(ts) // 1800}".encode()).hexdigest()[:16]
        for t, ts in zip(frame["token_address"], frame["candidate_timestamp"])
    ]
    # This file is an audit/public-attribute artifact; model input is a strict subset.
    lineage = {
        "decoder": "Relay(bytes32,address,address,address,uint256,uint256): discard word0 and word5; use words1-4",
        "source_files": [{"path": str(raw_path), "sha256": sha256_file(raw_path)}],
        "fields": {
            "candidate_tx_hash": {"source_file": str(raw_path), "location": "transactionHash", "rule": "public log identifier"},
            "candidate_timestamp": {"source_file": str(raw_path), "location": "blockTimestamp", "rule": "hex integer"},
            "amount_raw": {"source_file": str(raw_path), "location": "data word4", "rule": "uint256"},
            "receiver": {"source_file": str(raw_path), "location": "data word2", "rule": "ABI address"},
            "token_address": {"source_file": str(raw_path), "location": "data word3", "rule": "ABI address"},
            "block_number": {"source_file": str(raw_path), "location": "blockNumber", "rule": "hex integer"},
            "transaction_index": {"source_file": str(raw_path), "location": "transactionIndex", "rule": "hex integer"},
            "log_index": {"source_file": str(raw_path), "location": "logIndex", "rule": "hex integer"},
            "contract_address": {"source_file": str(raw_path), "location": "address", "rule": "public metadata"},
            "topic0": {"source_file": str(raw_path), "location": "topics[0]", "rule": "public event metadata"},
            "quotient_group_id": {"source_file": str(raw_path), "location": "token_address+candidate_timestamp", "rule": "deterministic audit grouping only"},
        },
        "discarded": {"data_word0": "protocol identity; never persisted", "data_word5": "unvalidated metadata; unused"},
        "label_sources_read": [],
        "access_boundary": {"max_timestamp_exclusive": max_timestamp_exclusive, "decoded_rows": len(rows), "embargo_rows_decoded": 0},
    }
    validate_inference_frame(frame[list(TARGET_ALLOWLIST)], side="target")
    return frame, lineage


def load_development_truth(path: Path, *, no_test_access: bool = True, split_manifest: bool = False) -> dict[str, str]:
    read_kwargs: dict[str, Any] = {"dtype": str}
    if no_test_access and split_manifest:
        manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
        read_kwargs["nrows"] = int(manifest["sample_counts"]["development"])
    df = pd.read_csv(path, **read_kwargs).fillna("")
    if "split" not in df.columns:
        raise AccessViolation("split column is required")
    if no_test_access and set(df["split"].str.lower()) - {"development"}:
        raise AccessViolation("development loader received embargo/test rows")
    mapping = dict(zip(df["source_tx_hash"].str.lower(), df["dest_tx_hash"].str.lower()))
    if len(mapping) < 1 or any(not k or not v for k, v in mapping.items()):
        raise AccessViolation("development truth mapping is empty or incomplete")
    return mapping


def _forbidden_match(name: str) -> bool:
    n = str(name).lower()
    return any(n == f or n.startswith(f) or f in n for f in FORBIDDEN_FIELDS)


def validate_inference_frame(frame: pd.DataFrame, *, side: str, field_lineage: dict[str, Any] | None = None) -> None:
    allowed = SOURCE_ALLOWLIST if side == "source" else TARGET_ALLOWLIST
    bad = [str(c) for c in frame.columns if str(c) not in allowed or _forbidden_match(str(c)) or str(c) in BRIDGE_FIELDS]
    if bad:
        raise ForbiddenInferenceField(f"Inference fields rejected for {side}: {sorted(bad)}")
    for field, meta in (field_lineage or {}).items():
        src = str((meta or {}).get("source_file") or "").lower()
        rule = str((meta or {}).get("rule") or "").lower()
        if "label" in src or "hash lookup" in rule or "ground truth" in rule or "oracle" in rule:
            raise ForbiddenInferenceField(f"Label-derived lineage rejected for {field}")


def evaluate_full_set(truth: dict[str, str], predictions: dict[str, str | None]) -> dict[str, Any]:
    tp = fp = fn = abstained = 0
    for source, target in truth.items():
        pred = predictions.get(source)
        if pred is None or pred == "":
            abstained += 1
            fn += 1
        elif _norm_hash(pred) == _norm_hash(target):
            tp += 1
        else:
            fp += 1
            fn += 1
    n = len(truth)
    predicted = n - abstained
    precision = tp / max(tp + fp, 1)
    recall = tp / max(n, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    coverage = predicted / max(n, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "n_total": n, "n_predicted": predicted, "n_abstained": abstained,
            "precision": precision, "recall": recall, "full_set_f1": f1, "coverage": coverage, "abstention": 1.0 - coverage}


def load_public_sources(split_path: Path, eth_path: Path, *, no_test_access: bool = True) -> pd.DataFrame:
    if no_test_access:
        manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
        development_count = int(manifest["sample_counts"]["development"])
        split = pd.read_csv(split_path, dtype=str, nrows=development_count).fillna("")
        required_split = "development"
    else:
        split = pd.read_csv(split_path, dtype=str).fillna("")
        required_split = "test"
    if "split" not in split.columns or set(split["split"].str.lower()) != {required_split}:
        raise AccessViolation(f"source loader requires {required_split}-only assignments")
    source_hashes = set(split["source_tx_hash"].map(_norm_hash))
    if not source_hashes:
        raise AccessViolation("no development source hashes")
    rows: list[pd.DataFrame] = []
    usecols = ["hash", "value", "timeStamp", "contractAddress"]
    for chunk in pd.read_csv(eth_path, dtype=str, usecols=usecols, chunksize=100000):
        chunk["source_tx_hash"] = chunk["hash"].map(_norm_hash)
        chunk = chunk[chunk["source_tx_hash"].isin(source_hashes)].copy()
        if not chunk.empty:
            rows.append(chunk)
    if not rows:
        raise AccessViolation("no public ETH sources found")
    df = pd.concat(rows, ignore_index=True)
    df["source_timestamp"] = pd.to_numeric(df["timeStamp"], errors="coerce").fillna(0).astype(int)
    df["source_amount_raw"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    df["source_token_address"] = df["contractAddress"].map(_norm_hash)
    df = df.sort_values(["source_tx_hash", "source_amount_raw"], ascending=[True, False]).drop_duplicates("source_tx_hash")
    df["source_route_type"] = df["source_token_address"].apply(classify_route_type_source)
    result = df[["source_tx_hash", "source_timestamp", "source_amount_raw", "source_token_address", "source_route_type"]].reset_index(drop=True)
    validate_inference_frame(result, side="source")
    return result


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _feature_frame(source: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = source.copy()
    t = candidates.copy()
    validate_inference_frame(s, side="source")
    validate_inference_frame(t[list(TARGET_ALLOWLIST)], side="target")
    return s, t


_PREDICT_CACHE: dict[str, tuple[dict[str, str | None], dict[str, float], dict[str, Any]]] = {}


def _normalize_mass(values: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(x.sum())
    return x / total if total > 0 else np.full(len(x), 1.0 / max(len(x), 1))


def _prediction_cache_key(source: pd.DataFrame, target: pd.DataFrame, config: dict[str, Any]) -> str:
    solver_config = {k: v for k, v in config.items() if k != "confidence_threshold"}
    fingerprint = {
        "config": solver_config,
        "source_rows": len(source),
        "target_rows": len(target),
        "source_time_sum": int(source["source_timestamp"].sum()),
        "target_time_sum": int(target["candidate_timestamp"].sum()),
    }
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()



def _make_quotient_key(row, strategy: str) -> str:
    if strategy == "token_30min":
        return hashlib.sha256(
            (str(row["token_address"]) + "|" + str(int(row["candidate_timestamp"]) // 1800)).encode()
        ).hexdigest()[:16]
    if strategy.startswith("contract_token_"):
        token = str(row.get("token_address", "")).lower()
        contract = str(row.get("contract_address", "")).lower()
        ts = int(row["candidate_timestamp"])
        if strategy.endswith("_ordermag"):
            amt = float(row.get("amount_raw", 0) or 0)
            mag = int(math.floor(math.log10(max(amt, 1e-12))))
            return hashlib.sha256(
                (contract + "|" + token + "|omag_" + str(mag)).encode()
            ).hexdigest()[:16]
        try:
            n_min = int(strategy.split("_")[-1].replace("min", ""))
        except (ValueError, IndexError):
            n_min = 30
        bucket = int(ts // (n_min * 60))
        return hashlib.sha256(
            (contract + "|" + token + "|bkt_" + str(bucket)).encode()
        ).hexdigest()[:16]
    return hashlib.sha256(
        (str(row["token_address"]) + "|" + str(int(row["candidate_timestamp"]) // 1800)).encode()
    ).hexdigest()[:16]


def predict(source: pd.DataFrame, candidates: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, str | None], dict[str, Any]]:
    """Run RC-UOT-Q using only independently observed generic chain features."""
    s, t = _feature_frame(source, candidates)
    key = _prediction_cache_key(s, t, config)
    cached = _PREDICT_CACHE.get(key)
    if cached is None:
        before = float(config["before_sec"])
        after = float(config["window_h"]) * 3600.0
        aw, tw, kw = float(config["amount_weight"]), float(config["time_weight"]), float(config["token_weight"])
        rw = float(config.get("route_weight", 0.0))
        total = max(aw + tw + kw + rw, 1e-12)
        aw, tw, kw, rw = aw / total, tw / total, kw / total, rw / total
        top_k = int(config.get("top_k", 20))
        batch_size = int(config.get("batch_size", 50))
        use_transport = bool(config.get("use_transport", True))
        raw_predictions: dict[str, str | None] = {}
        confidence: dict[str, float] = {}
        candidate_sets: list[list[int]] = []
        for row in s.itertuples(index=False):
            ct = t[(t["candidate_timestamp"] >= row.source_timestamp - before) & (t["candidate_timestamp"] <= row.source_timestamp + after)]
            if ct.empty or not np.isfinite(float(row.source_amount_raw)) or not str(row.source_token_address):
                candidate_sets.append([])
                continue
            amounts = pd.to_numeric(ct["amount_raw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            amount_cost = np.minimum(np.abs(amounts - float(row.source_amount_raw)) / np.maximum(np.maximum(amounts, float(row.source_amount_raw)), 1e-12), 1.0)
            delay = ct["candidate_timestamp"].to_numpy(dtype=float) - float(row.source_timestamp)
            time_cost = np.where(delay < 0, 1.0 + np.minimum(np.abs(delay) / max(after, 1.0), 1.0), np.minimum(delay / max(after, 1.0), 1.0))
            token_cost = (ct["token_address"].fillna("").str.lower().to_numpy() != str(row.source_token_address).lower()).astype(float)
            src_rt = str(getattr(row, "source_route_type", "unknown")).lower()
            route_cost_item = (ct["route_type"].fillna("unknown").str.lower().to_numpy() != src_rt).astype(float)
            heuristic = aw * amount_cost + tw * time_cost + kw * token_cost + rw * route_cost_item
            order = np.lexsort((ct["candidate_tx_hash"].astype(str).to_numpy(), heuristic))[:top_k]
            candidate_sets.append(ct.index.to_numpy(dtype=int)[order].tolist())
        for start in range(0, len(s), batch_size):
            stop = min(start + batch_size, len(s))
            src_batch = s.iloc[start:stop]
            original_indices = sorted({j for i in range(start, stop) for j in candidate_sets[i]})
            if not original_indices:
                for row in src_batch.itertuples(index=False):
                    raw_predictions[row.source_tx_hash] = None
                    confidence[row.source_tx_hash] = 0.0
                continue
            dst_batch = t.iloc[original_indices].reset_index(drop=True)
            grouping_strategy = str(config.get("grouping_strategy", "token_30min"))
            if grouping_strategy != "none":
                group_key = [_make_quotient_key(dst_batch.iloc[i], grouping_strategy) for i in range(len(dst_batch))]
            else:
                group_key = dst_batch["candidate_tx_hash"].astype(str).tolist()
            dst_batch = dst_batch.assign(_quotient_key=group_key)
            grouped = dst_batch.groupby("_quotient_key", sort=True, dropna=False)
            quotient_rows = grouped.agg(amount_raw=("amount_raw", "sum"), candidate_timestamp=("candidate_timestamp", "min"), token_address=("token_address", "first"), route_type=("route_type", "first")).reset_index()
            member_map = {name: list(group.index) for name, group in grouped}
            group_names = quotient_rows["_quotient_key"].astype(str).tolist()
            local_to_group = {member: gi for gi, name in enumerate(group_names) for member in member_map[name]}
            original_to_local = {original: local for local, original in enumerate(original_indices)}
            src_amount = src_batch["source_amount_raw"].to_numpy(dtype=float)
            dst_amount = quotient_rows["amount_raw"].to_numpy(dtype=float)
            src_time = src_batch["source_timestamp"].to_numpy(dtype=float)
            dst_time = quotient_rows["candidate_timestamp"].to_numpy(dtype=float)
            amount_cost = np.minimum(np.abs(src_amount[:, None] - dst_amount[None, :]) / np.maximum(np.maximum(src_amount[:, None], dst_amount[None, :]), 1e-12), 1.0)
            delay = dst_time[None, :] - src_time[:, None]
            time_cost = np.where(delay < 0, 1.0 + np.minimum(np.abs(delay) / max(after, 1.0), 1.0), np.minimum(delay / max(after, 1.0), 1.0))
            token_cost = (quotient_rows["token_address"].fillna("").str.lower().to_numpy()[None, :] != src_batch["source_token_address"].fillna("").str.lower().to_numpy()[:, None]).astype(float)
            src_rt_arr = src_batch["source_route_type"].fillna("unknown").str.lower().to_numpy()
            quotient_rt = quotient_rows["route_type"].fillna("unknown").str.lower().to_numpy()
            route_cost_mat = (quotient_rt[None, :] != src_rt_arr[:, None]).astype(float)
            cost = aw * amount_cost + tw * time_cost + kw * token_cost + rw * route_cost_mat
            if use_transport:
                plan = uot_sinkhorn(_normalize_mass(src_amount), _normalize_mass(dst_amount), cost, epsilon=float(config.get("reg", 0.05)), tau=float(config.get("reg_m", 0.5)), max_iter=int(config.get("max_iter", 100)), tol=float(config.get("tol", 1e-6)))
            else:
                plan = np.exp(-cost / max(float(config.get("reg", 0.05)), 1e-12))
            for local_i, row in enumerate(src_batch.itertuples(index=False)):
                source_original = set(candidate_sets[start + local_i])
                allowed_local = [original_to_local[x] for x in source_original if x in original_to_local]
                allowed_groups = {local_to_group[x] for x in allowed_local}
                row_plan = plan[local_i].copy()
                row_plan[delay[local_i] < 0] = 0.0
                row_plan[[j for j in range(len(row_plan)) if j not in allowed_groups]] = 0.0
                if not allowed_groups or float(row_plan.sum()) <= 0:
                    raw_predictions[row.source_tx_hash] = None
                    confidence[row.source_tx_hash] = 0.0
                    continue
                best_group = int(np.argmax(row_plan))
                group_confidence = float(row_plan[best_group] / max(float(row_plan.sum()), 1e-12))
                members = [x for x in member_map[group_names[best_group]] if x in allowed_local]
                if not members:
                    raw_predictions[row.source_tx_hash] = None
                    confidence[row.source_tx_hash] = 0.0
                    continue
                member_df = dst_batch.iloc[members]
                member_amount = member_df["amount_raw"].to_numpy(dtype=float)
                local_amount = np.minimum(np.abs(member_amount - float(row.source_amount_raw)) / np.maximum(np.maximum(member_amount, float(row.source_amount_raw)), 1e-12), 1.0)
                local_delay = member_df["candidate_timestamp"].to_numpy(dtype=float) - float(row.source_timestamp)
                local_time = np.where(local_delay < 0, 2.0, np.minimum(local_delay / max(after, 1.0), 1.0))
                local_token = (member_df["token_address"].fillna("").str.lower().to_numpy() != str(row.source_token_address).lower()).astype(float)
                local_rt = (member_df["route_type"].fillna("unknown").str.lower().to_numpy() != str(getattr(row, "source_route_type", "unknown")).lower()).astype(float)
                local_cost = aw * local_amount + tw * local_time + kw * local_token + rw * local_rt
                local_order = np.lexsort((member_df["candidate_tx_hash"].astype(str).to_numpy(), local_cost))
                best_member = members[int(local_order[0])]
                raw_predictions[row.source_tx_hash] = str(dst_batch.iloc[best_member]["candidate_tx_hash"])
                confidence[row.source_tx_hash] = group_confidence * float(np.exp(-local_cost[int(local_order[0])]))
        diagnostics = {
            "transport": "cross.domain.uot.uot_solver_numpy.uot_sinkhorn" if use_transport else "disabled_ablation",
            "grouping_strategy": grouping_strategy,
            "coverage_qualification": "public fields complete and nonempty candidate set",
            "top_k": top_k,
            "route_weight": rw,
        }
        cached = (raw_predictions, confidence, diagnostics)
        _PREDICT_CACHE[key] = cached
    raw_predictions, confidence, diagnostics = cached
    threshold = float(config["confidence_threshold"])
    predictions = {source_id: (target_id if target_id is not None and confidence.get(source_id, 0.0) >= threshold else None) for source_id, target_id in raw_predictions.items()}
    return predictions, {"confidence": confidence, **diagnostics}

def candidate_recall(truth: dict[str, str], candidates: pd.DataFrame, *, before_sec: float, after_sec: float) -> dict[str, Any]:
    ids = set(candidates["candidate_tx_hash"].map(_norm_hash))
    covered = sum(_norm_hash(d) in ids for d in truth.values())
    return {"n_truth": len(truth), "n_candidate_transactions": len(ids), "covered": covered, "candidate_pool_recall": covered / max(len(truth), 1), "missed": len(truth) - covered}


def run_heldout_once(freeze_path: Path, authorization_path: Path, output_path: Path, evaluator: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    if not freeze_path.is_file() or not authorization_path.is_file():
        raise HeldoutGateError("freeze manifest and authorization file are required")
    manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
    auth = json.loads(authorization_path.read_text(encoding="utf-8"))
    if manifest.get("test_run_counter") != 0 or output_path.exists():
        raise HeldoutGateError("held-out run is already consumed")
    if auth.get("schema_version") != 1 or auth.get("authorized") is not True or auth.get("freeze_id") != manifest.get("freeze_id") or not auth.get("authorization_id"):
        raise HeldoutGateError("authorization does not match frozen manifest")
    lock_path = freeze_path.parent / "heldout_run.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise HeldoutGateError("held-out run lock already exists") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as lock:
            json.dump({"freeze_id": manifest["freeze_id"], "authorization_id": auth["authorization_id"], "state": "RUNNING"}, lock)
            lock.flush()
            os.fsync(lock.fileno())
        result = evaluator(auth)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(output_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
        manifest["test_run_counter"] = 1
        manifest["heldout_output_exists"] = True
        atomic_write_text(freeze_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        atomic_write_text(lock_path, json.dumps({"freeze_id": manifest["freeze_id"], "state": "CONSUMED"}, sort_keys=True) + "\n")
    except Exception:
        if not output_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass
        raise
