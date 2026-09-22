#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bootstrap CLI with ``src`` on sys.path (editable / from repo root)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cross.application.bootstrap import main
from cross.domain.path_a.service import path_a_vs_label_cmp as _path_a_vs_label_cmp


def path_a_vs_label_cmp(pairs_df: pd.DataFrame, label_path: Path) -> dict:
    """Backward-compatible export for existing tests/importers."""
    return _path_a_vs_label_cmp(pairs_df, label_path)


if __name__ == "__main__":
    import logging
    import sys

    _log = logging.getLogger(__name__)
    try:
        _code = main()
    except SystemExit as _e:
        raise _e
    except BaseException:
        _log.exception("Pipeline failed")
        sys.exit(1)
    raise SystemExit(_code)
