from decimal import Decimal

from cross.shared.amount_normalizer import expected_raw_ratio, raw_to_human


def test_expected_raw_ratio_stablecoin_style():
    r = expected_raw_ratio(6, 18, Decimal("1"))
    assert r == Decimal(10) ** 12


def test_raw_to_human():
    h = raw_to_human(10**6, 6)
    assert h == Decimal(1)
