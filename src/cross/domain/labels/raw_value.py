"""Parse EVM ``value`` fields as integers (no float); optional Decimal for exact arithmetic."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any


_HEX_RE = re.compile(r"^0x[0-9a-fA-F]+$")


def parse_raw_value(raw: Any) -> tuple[int | None, str]:
    """Parse ``value`` from CSV / JSON (hex ``0x...`` or decimal string). Returns ``(int_or_none, invalid_reason)``."""
    if raw is None or (isinstance(raw, float) and str(raw) == "nan"):
        return None, "null_value"
    s = str(raw).strip()
    if not s:
        return None, "empty_string"
    try:
        if _HEX_RE.match(s):
            v = int(s, 16)
            return v, ""
        if s.startswith("0x") or s.startswith("0X"):
            v = int(s, 16)
            return v, ""
        v = int(s, 10)
        return v, ""
    except (ValueError, OverflowError):
        try:
            d = Decimal(s)
            if d != d.to_integral_value():
                return None, "fractional_not_allowed"
            iv = int(d)
            return iv, ""
        except (InvalidOperation, ValueError, OverflowError):
            return None, "unparseable"


def raw_value_decimal(raw_int: int | None) -> Decimal | None:
    if raw_int is None:
        return None
    try:
        return Decimal(raw_int)
    except Exception:
        return None
