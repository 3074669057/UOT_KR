from pathlib import Path

from cross.domain.path_b.route_registry import RouteRegistry
from cross.shared.decimals_registry import DecimalsRegistry


def test_route_registry_usdt_ratio():
    routes_path = Path(__file__).resolve().parents[1] / "config" / "token_routes.eth_bsc.json"
    rr = RouteRegistry.load(routes_path, decimals_registry=DecimalsRegistry())
    assert rr is not None
    pair = rr.ratio_by_eth_bnb_pair(None)
    eth_usdt = "0xdac17f958d2ee523a2206206994597c13d831ec7"
    bsc_usdt = "0x55d398326f99059ff775485246999027b3197955"
    assert pair.get((eth_usdt, bsc_usdt)) == float(10**12)
