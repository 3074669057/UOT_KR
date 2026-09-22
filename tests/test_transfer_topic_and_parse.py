"""ERC20 Transfer topic constant and parse_transfer_log."""
from __future__ import annotations

from cross.infrastructure.online import bnb_window_fetch as bwf
from cross.infrastructure.online.bnb_window_fetch import TRANSFER_TOPIC, parse_transfer_log

CANONICAL_TRANSFER_TOPIC = (
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
)


def test_transfer_topic_is_canonical_erc20() -> None:
    assert TRANSFER_TOPIC.lower() == CANONICAL_TRANSFER_TOPIC.lower()


def test_parse_transfer_log_success() -> None:
    log = {
        "address": "0x55d398326f99059ff775485246999027b3197955",
        "topics": [
            CANONICAL_TRANSFER_TOPIC,
            "0x000000000000000000000000aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "0x000000000000000000000000bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        ],
        "data": "0x01",
        "transactionHash": "0x" + "cc" * 32,
        "blockNumber": "0xa",
        "logIndex": "0x0",
    }
    out = parse_transfer_log(log)
    assert out is not None
    assert out["token_contract"] == "0x55d398326f99059ff775485246999027b3197955"
    assert out["raw_amount"] == 1
    assert out["evidence_level"] == "token_transfer_log"


def test_hex_to_int_bare_0x() -> None:
    """RPC may return bare ``0x`` for empty quantity; must not raise."""
    assert bwf._hex_to_int("0x") == 0
    assert bwf._hex_to_int("") == 0
    assert bwf._hex_to_int("0x0a") == 10
    assert bwf._hex_to_int("0x0") == 0


def test_parse_transfer_log_wrong_topic() -> None:
    log = {
        "address": "0x55d398326f99059ff775485246999027b3197955",
        "topics": ["0x" + "00" * 32],
        "data": "0x0",
        "transactionHash": "0x" + "dd" * 32,
    }
    assert parse_transfer_log(log) is None
