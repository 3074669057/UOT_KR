"""Ethereum-style transaction hash normalization (32-byte hex), distinct from 20-byte addresses."""
from __future__ import annotations

import re

_TX_BODY = re.compile(r"^[0-9a-f]{64}$")


def normalize_tx_hash(raw: object) -> str:
    """Return canonical ``0x`` + 64 lowercase hex, or empty string if invalid.

    Accepts optional ``0x`` prefix. Does **not** use address helpers (20-byte).
    """
    if raw is None or (isinstance(raw, float) and str(raw) == "nan"):
        return ""
    s = str(raw).strip().lower()
    if not s or s == "nan":
        return ""
    if s.startswith("0x"):
        body = s[2:]
    else:
        body = s
    if not _TX_BODY.match(body):
        return ""
    return "0x" + body


def validate_tx_hash(raw: object) -> bool:
    """True iff ``raw`` is a valid 32-byte tx hash string after normalization."""
    return bool(normalize_tx_hash(raw))
