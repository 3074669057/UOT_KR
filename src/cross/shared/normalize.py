def norm_addr(addr: object) -> str:
    """Lowercase hex address string; empty if missing."""
    if addr is None or (isinstance(addr, float)):
        return ""
    s = str(addr).strip().lower()
    if not s or s == "nan":
        return ""
    return s if s.startswith("0x") else "0x" + s
