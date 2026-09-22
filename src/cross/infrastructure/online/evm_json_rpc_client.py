"""Generic EVM chain JSON-RPC client (fixed URLs or endpoint template + API keys)."""
from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvmJsonRpcClient:
    """Minimal JSON-RPC over HTTP with URL rotation and optional extra headers."""

    urls: list[str]
    timeout_sec: float = 20.0
    max_retries_per_key: int = 2
    extra_headers: dict[str, str] = field(default_factory=dict)
    _idx: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.urls = [str(u).strip() for u in self.urls if str(u).strip()]
        if not self.urls:
            raise ValueError("EvmJsonRpcClient requires non-empty urls")
        self.extra_headers = {
            str(k): str(v) for k, v in (self.extra_headers or {}).items() if str(k).strip()
        }

    @classmethod
    def from_template(
        cls,
        endpoint_template: str,
        api_keys: list[str],
        *,
        timeout_sec: float = 20.0,
        max_retries_per_key: int = 2,
        extra_headers: dict[str, str] | None = None,
    ) -> EvmJsonRpcClient:
        tmpl = str(endpoint_template or "").strip()
        keys = [str(k).strip() for k in api_keys if str(k).strip()]
        if not tmpl or "{api_key}" not in tmpl:
            raise ValueError("endpoint_template must contain {api_key} placeholder")
        if not keys:
            raise ValueError("from_template requires non-empty api_keys")
        urls = [tmpl.replace("{api_key}", k) for k in keys]
        return cls(
            urls=urls,
            timeout_sec=float(timeout_sec),
            max_retries_per_key=int(max_retries_per_key),
            extra_headers=dict(extra_headers or {}),
        )

    @classmethod
    def from_urls(
        cls,
        urls: list[str],
        *,
        timeout_sec: float = 20.0,
        max_retries_per_key: int = 2,
        extra_headers: dict[str, str] | None = None,
    ) -> EvmJsonRpcClient:
        u = [str(x).strip() for x in urls if str(x).strip()]
        if not u:
            raise ValueError("from_urls requires non-empty rpc_urls")
        return cls(
            urls=u,
            timeout_sec=float(timeout_sec),
            max_retries_per_key=int(max_retries_per_key),
            extra_headers=dict(extra_headers or {}),
        )

    def _merge_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", **self.extra_headers}

    def _rpc_once(self, url: str, method: str, params: list[Any]) -> Any:
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST", headers=self._merge_headers())
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"rpc error: {data['error']}")
        return data.get("result") if isinstance(data, dict) else None

    def _rpc_batch_once(self, url: str, calls: list[tuple[str, list[Any]]]) -> list[Any]:
        body = json.dumps(
            [
                {"jsonrpc": "2.0", "id": idx + 1, "method": method, "params": params}
                for idx, (method, params) in enumerate(calls)
            ]
        ).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST", headers=self._merge_headers())
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not isinstance(data, list):
            raise RuntimeError("rpc batch error: non-list response")
        by_id: dict[int, Any] = {}
        for item in data:
            if isinstance(item, dict) and item.get("error"):
                raise RuntimeError(f"rpc batch error: {item['error']}")
            try:
                rid = int(item.get("id"))
            except Exception:
                continue
            by_id[rid] = item.get("result")
        out: list[Any] = []
        for idx in range(len(calls)):
            out.append(by_id.get(idx + 1))
        return out

    def rpc(self, method: str, params: list[Any]) -> Any:
        n = len(self.urls)
        last_err: Exception | None = None
        for shift in range(n):
            url = self.urls[(self._idx + shift) % n]
            for _ in range(max(int(self.max_retries_per_key), 1)):
                try:
                    out = self._rpc_once(url, method, params)
                    self._idx = (self._idx + shift + 1) % n
                    return out
                except (
                    urllib.error.URLError,
                    urllib.error.HTTPError,
                    TimeoutError,
                    RuntimeError,
                    ValueError,
                    http.client.IncompleteRead,
                ) as e:
                    last_err = e
                    continue
        raise RuntimeError(f"JSON-RPC failed after URL rotation: {method}") from last_err

    def rpc_batch(self, calls: list[tuple[str, list[Any]]]) -> list[Any]:
        if not calls:
            return []
        n = len(self.urls)
        last_err: Exception | None = None
        for shift in range(n):
            url = self.urls[(self._idx + shift) % n]
            for _ in range(max(int(self.max_retries_per_key), 1)):
                try:
                    out = self._rpc_batch_once(url, calls)
                    self._idx = (self._idx + shift + 1) % n
                    return out
                except (
                    urllib.error.URLError,
                    urllib.error.HTTPError,
                    TimeoutError,
                    RuntimeError,
                    ValueError,
                    http.client.IncompleteRead,
                ) as e:
                    last_err = e
                    continue
        raise RuntimeError("JSON-RPC batch failed after URL rotation") from last_err

    def get_block_by_number(self, block_num: int, *, full_transactions: bool = True) -> Any:
        """Return block object for ``eth_getBlockByNumber`` (``full_transactions`` = full tx dicts)."""
        bn = max(int(block_num), 0)
        return self.rpc("eth_getBlockByNumber", [hex(bn), full_transactions])

    def get_transaction_receipt(self, tx_hash: str) -> Any:
        """Return receipt dict or ``None`` for ``eth_getTransactionReceipt``."""
        h = str(tx_hash or "").strip()
        if not h.startswith("0x") and len(h) == 64:
            h = "0x" + h
        return self.rpc("eth_getTransactionReceipt", [h])

    def get_transaction_receipts_batch(
        self,
        tx_hashes: list[str],
        *,
        chunk_size: int = 80,
    ) -> list[Any]:
        """Fetch receipts via JSON-RPC batch; falls back to sequential ``get_transaction_receipt`` on batch failure.

        Returns a list aligned with ``tx_hashes`` (``None`` if receipt missing).
        """
        out: list[Any] = []
        raw_hashes = [str(h or "").strip() for h in tx_hashes]
        step = max(int(chunk_size), 1)
        for i in range(0, len(raw_hashes), step):
            chunk = raw_hashes[i : i + step]
            calls: list[tuple[str, list[Any]]] = []
            for th in chunk:
                h = th if th.startswith("0x") else ("0x" + th if len(th) == 64 else th)
                calls.append(("eth_getTransactionReceipt", [h]))
            try:
                batch_out = self.rpc_batch(calls)
                for r in batch_out:
                    out.append(r)
            except Exception:
                for th in chunk:
                    try:
                        out.append(self.get_transaction_receipt(th))
                    except Exception:
                        out.append(None)
        return out


def client_from_runtime_block(nr: dict[str, Any]) -> EvmJsonRpcClient:
    """Build a client from a ``nodereal`` / ``ethereum`` JSON config fragment."""
    timeout_sec = float(nr.get("timeout_sec", 20))
    retries = int(nr.get("max_retries_per_key", 2))
    headers_raw = nr.get("rpc_headers") or {}
    headers: dict[str, str] = (
        {str(k): str(v) for k, v in headers_raw.items()}
        if isinstance(headers_raw, dict)
        else {}
    )

    urls_cfg = nr.get("rpc_urls") or nr.get("rpc_url")
    urls: list[str] = []
    if isinstance(urls_cfg, str) and str(urls_cfg).strip():
        urls = [str(urls_cfg).strip()]
    elif isinstance(urls_cfg, list):
        urls = [str(u).strip() for u in urls_cfg if str(u).strip()]

    if urls:
        return EvmJsonRpcClient.from_urls(
            urls,
            timeout_sec=timeout_sec,
            max_retries_per_key=retries,
            extra_headers=headers,
        )

    keys = [str(k).strip() for k in list(nr.get("api_keys") or []) if str(k).strip()]
    tmpl = str(nr.get("endpoint_template") or "").strip()
    if not keys:
        raise ValueError("rpc client requires non-empty api_keys or rpc_urls")
    return EvmJsonRpcClient.from_template(
        tmpl,
        keys,
        timeout_sec=timeout_sec,
        max_retries_per_key=retries,
        extra_headers=headers,
    )


__all__ = ["EvmJsonRpcClient", "client_from_runtime_block"]
