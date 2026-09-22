"""Minimal ETH mainnet JSON-RPC helpers (tx + block timestamp). Optional for two-stage."""
from __future__ import annotations

from cross.shared.normalize import norm_addr

from .evm_json_rpc_client import EvmJsonRpcClient


def _hex_to_int(x: str | None) -> int:
    s = str(x or "0x0")
    return int(s, 16) if s.startswith("0x") else int(s)


def fetch_eth_tx_timestamp(client: EvmJsonRpcClient, txhash: str) -> int | None:
    """Return block unix timestamp for ``txhash``, or None if not found."""
    txh = norm_addr(txhash)
    tx = client.rpc("eth_getTransactionByHash", [txh]) or {}
    bn = tx.get("blockNumber")
    if not bn:
        return None
    block_num = _hex_to_int(bn)
    blk = client.rpc("eth_getBlockByNumber", [hex(block_num), False]) or {}
    return _hex_to_int(blk.get("timestamp"))
