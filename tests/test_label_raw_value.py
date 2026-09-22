from cross.domain.labels.raw_value import parse_raw_value


def test_parse_hex_zero():
    v, err = parse_raw_value("0x0")
    assert err == ""
    assert v == 0


def test_parse_decimal():
    v, err = parse_raw_value("1000")
    assert err == ""
    assert v == 1000


def test_parse_hex_nonzero():
    v, err = parse_raw_value("0x10")
    assert err == ""
    assert v == 16
