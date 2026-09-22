"""NaN-safe scalar coercion for pandas, numpy, and RPC-style string inputs."""
from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[misc, assignment]

_HEX_INT_RE = re.compile(r"^0x[0-9a-fA-F]+$", re.IGNORECASE)


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if value is pd.NA:
        return True
    try:
        if pd.api.types.is_scalar(value) and pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and math.isnan(value):
        return True
    if np is not None:
        try:
            if isinstance(value, np.generic):
                if np.issubdtype(value.dtype, np.floating) and np.isnan(value):
                    return True
        except (TypeError, ValueError):
            pass
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def first_non_null(*values: Any, default: Any = None) -> Any:
    for v in values:
        if not _is_nullish(v):
            return v
    return default


def _parse_hex_int_token(s: str) -> int | None:
    st = s.strip()
    if not st.lower().startswith("0x"):
        return None
    body = st[2:]
    if not body:
        return 0
    if not _HEX_INT_RE.match(st):
        return None
    try:
        return int(body, 16)
    except ValueError:
        return None


def safe_int(value: Any, default: int = 0) -> int:
    if _is_nullish(value):
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        st = value.strip()
        if not st:
            return default
        hx = _parse_hex_int_token(st)
        if hx is not None:
            return hx
    try:
        num = pd.to_numeric(value, errors="coerce")
    except Exception:
        return default
    if isinstance(num, pd.Series):
        num = num.iloc[0] if len(num.index) else float("nan")
    try:
        if pd.isna(num):
            return default
    except (TypeError, ValueError):
        pass
    try:
        xf = float(num)
    except (TypeError, ValueError):
        return default
    if math.isnan(xf):
        return default
    try:
        return int(xf)
    except (ValueError, OverflowError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    if _is_nullish(value):
        return default
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, str):
        st = value.strip()
        if not st:
            return default
        hx = _parse_hex_int_token(st)
        if hx is not None:
            return float(hx)
    try:
        num = pd.to_numeric(value, errors="coerce")
    except Exception:
        return default
    if isinstance(num, pd.Series):
        num = num.iloc[0] if len(num.index) else float("nan")
    try:
        if pd.isna(num):
            return default
    except (TypeError, ValueError):
        pass
    try:
        xf = float(num)
    except (TypeError, ValueError):
        return default
    if math.isnan(xf):
        return default
    return xf


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if value is pd.NA:
        return default
    try:
        if pd.api.types.is_scalar(value) and pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and math.isnan(value):
        return default
    if np is not None:
        try:
            if isinstance(value, np.generic) and np.issubdtype(value.dtype, np.floating) and np.isnan(value):
                return default
        except (TypeError, ValueError):
            pass
    if isinstance(value, str):
        return value if value else default
    return str(value)


def safe_bool(value: Any, default: bool = False) -> bool:
    if _is_nullish(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off", ""):
            return False
    try:
        num = pd.to_numeric(value, errors="coerce")
    except Exception:
        return bool(value) if value is not None else default
    if isinstance(num, pd.Series):
        num = num.iloc[0] if len(num.index) else float("nan")
    try:
        if pd.isna(num):
            return default
    except (TypeError, ValueError):
        pass
    try:
        xf = float(num)
    except (TypeError, ValueError):
        return default
    if math.isnan(xf):
        return default
    return xf != 0.0


__all__ = ["first_non_null", "safe_bool", "safe_float", "safe_int", "safe_str"]
