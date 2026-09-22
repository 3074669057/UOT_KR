"""Deprecated: root logging is configured in ``cross.application.bootstrap`` via ``run_init.setup_logging``."""
from __future__ import annotations


def configure_root_logging_once(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
    """No-op. Do not add handlers from domain modules; use the CLI/bootstrap entry path."""
    return
