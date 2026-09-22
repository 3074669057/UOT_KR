"""Detect whether a Celer-style label CSV exists and has at least one row."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cross.shared.load_csv import load_label_csv


def label_file_usable(path: Path | str) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    try:
        df = load_label_csv(p)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return False
    except ValueError:
        return False
    return len(df) > 0
