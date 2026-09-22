"""Environment toggles read by ``WithdrawLocator.search_withdraw``."""
from __future__ import annotations

import os


def configure_path_b_connector_env(
    *,
    fee_threshold: float = 0.12,
    time_gap_seconds: float = 10800.0,
) -> None:
    """Set ``CONNECTOR_*`` env vars so WithdrawLocator rules match Path B tuning.

    Thresholds are read inside ``WithdrawLocator.search_withdraw`` each time via env.
    """
    os.environ["CONNECTOR_FEE_THRESHOLD"] = str(fee_threshold)
    os.environ["CONNECTOR_TIME_THRESHOLD"] = str(time_gap_seconds)
