"""One-shot output directory prep and root logging setup (process entry only)."""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from cross.config.output_layout import output_file
from cross.config.paths import CROSS_ROOT

_PROTECTED_DIR_NAMES = frozenset(
    {"in", "input", "config", "label", "labels", "src", "data", "tests", "scripts"}
)


def _parse_log_level(name: str | None, default: int = logging.INFO) -> int:
    if not name or not str(name).strip():
        return default
    n = str(name).strip().upper()
    return int(getattr(logging, n, default))


def _is_filesystem_root(p: Path) -> bool:
    r = p.resolve()
    s = str(r)
    if r.anchor and s == r.anchor:
        return True
    if s in ("/", "//"):
        return True
    # Windows drive root: resolved path has a single top-level component
    if r.drive and len(r.parts) == 1:
        return True
    return False


def prepare_output_dir(
    out_dir: str | Path,
    *,
    clear: bool = True,
    project_root: Path | None = None,
    config_output_dir: str | None = None,
) -> Path:
    """Ensure ``out_dir`` exists; optionally delete prior run artifacts inside it only."""
    project_root = (project_root or CROSS_ROOT).resolve()
    raw = Path(out_dir).expanduser()
    if not str(raw).strip():
        raise ValueError("output dir is empty")

    out_path = raw.resolve() if raw.is_absolute() else (project_root / raw).resolve()

    if not str(out_path):
        raise ValueError("output dir is empty")

    if out_path == project_root:
        raise ValueError(f"Refusing to clear project root: {out_path}")

    if _is_filesystem_root(out_path):
        raise ValueError(f"Refusing to clear filesystem root: {out_path}")

    if out_path.name.lower() in _PROTECTED_DIR_NAMES:
        raise ValueError(f"Refusing to clear protected directory: {out_path}")

    name_ok = any(p.lower() == "out" for p in out_path.parts) or "out" in out_path.name.lower()
    cfg_ok = False
    if config_output_dir and str(config_output_dir).strip():
        cfg_cand = (project_root / Path(str(config_output_dir).strip())).resolve()
        cfg_ok = out_path == cfg_cand
    if not name_ok and not cfg_ok:
        raise ValueError(f"Refusing to clear suspicious output dir without 'out' in name: {out_path}")

    if clear and out_path.exists():
        for child in out_path.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=False)
            else:
                child.unlink()

    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def setup_logging(out_dir: Path, cfg: dict[str, Any] | None = None, *, level: int | None = None) -> Path:
    """Attach fresh root handlers: one UTF-8 ``run.log`` (overwrite) and one console stream."""
    cfg = cfg or {}
    log_to_file = bool(cfg.get("log_to_file", True))
    log_name = str(cfg.get("log_file_name") or "run.log").strip() or "run.log"
    file_level = _parse_log_level(str(cfg.get("file_log_level") or "INFO"), logging.INFO)
    console_level = _parse_log_level(str(cfg.get("console_log_level") or "INFO"), logging.INFO)
    if level is not None:
        root_level = int(level)
    elif log_to_file:
        root_level = int(min(file_level, console_level))
    else:
        root_level = int(console_level)

    log_file = output_file(Path(out_dir), log_name) if log_name == "run.log" else Path(out_dir) / log_name

    root = logging.getLogger()
    root.setLevel(root_level)

    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    file_formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    console_formatter = logging.Formatter("%(levelname)s %(message)s")

    if log_to_file:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    return log_file
