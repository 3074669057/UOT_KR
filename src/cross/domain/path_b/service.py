from __future__ import annotations

from pathlib import Path

import pandas as pd

from .runner import run_path_b, write_path_b_outputs


def execute_path_b(**kwargs):
    return run_path_b(**kwargs)


def persist_path_b_outputs(
    out_dir: Path,
    pairs,
    cmp_dict: dict,
    *,
    eth_path: Path,
    bnb_df: pd.DataFrame,
    validate_evidence_schema: bool = False,
) -> None:
    write_path_b_outputs(
        out_dir,
        pairs,
        cmp_dict,
        eth_path=eth_path,
        bnb_df=bnb_df,
        write_address_mapping=True,
        validate_evidence_schema=validate_evidence_schema,
    )
