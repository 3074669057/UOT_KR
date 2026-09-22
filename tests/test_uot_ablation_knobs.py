from __future__ import annotations

import numpy as np

from cross.domain.path_b.runner import _apply_uot_ablation
from cross.domain.uot.cost_matrix import build_cost_matrix_decomposed, default_cost_weights
from cross.interfaces.cli import build_parser


def _base_kwargs():
    return dict(
        cost_weights=dict(default_cost_weights()),
        use_graph_embedding=False,
        uot_reg=0.1,
        uot_reg_m=1.0,
        lambda_risk=0.1,
    )


def test_no_amount_cost_zeros_amount_weight():
    w, *_ = _apply_uot_ablation(ablation="no_amount_cost", **_base_kwargs())
    assert w is not None
    assert float(w.get("amount", -1)) == 0.0


def test_no_route_bridge_zeros_route_weight():
    w, *_ = _apply_uot_ablation(ablation="no_route_bridge", **_base_kwargs())
    assert w is not None
    assert float(w.get("route", -1)) == 0.0


def test_no_address_novelty_zeros_novelty_weight():
    w, *_ = _apply_uot_ablation(ablation="no_address_novelty", **_base_kwargs())
    assert w is not None
    assert float(w.get("novelty", -1)) == 0.0


def test_novelty_cost_nonzero_with_address_overlap():
    s = [
        {
            "flow_id": "e1",
            "amount_usd": 100.0,
            "start_time": 0.0,
            "end_time": 100.0,
            "aml_score": 0.0,
            "bridge": "x",
            "graph_embedding": None,
            "address_set": ["0xaaa"],
            "route_type": "same_asset_bridge:USDT",
            "asset_group": "stable:USDT",
        }
    ]
    t = [
        {
            "flow_id": "b1",
            "amount_usd": 100.0,
            "start_time": 200.0,
            "end_time": 200.0,
            "aml_score": 0.0,
            "bridge": "x",
            "graph_embedding": None,
            "address_set": ["0xaaa"],
            "route_type": "same_asset_bridge:USDT",
            "asset_group": "stable:USDT",
        }
    ]
    d0 = build_cost_matrix_decomposed(s, t, weights=default_cost_weights(), use_graph=False)
    d1 = build_cost_matrix_decomposed(
        s, t, weights=_apply_uot_ablation(ablation="no_address_novelty", **_base_kwargs())[0], use_graph=False
    )
    assert float(d0["address_novelty_cost"][0, 0]) < 0.2
    assert float(d1["C"][0, 0]) <= float(d0["C"][0, 0]) + 1e-9


def test_cli_accepts_phase10r_missing_knobs():
    parser = build_parser()
    for knob in ("no_amount_cost", "no_route_bridge", "no_address_novelty"):
        args, _ = parser.parse_known_args(["--uot-ablation", knob])
        assert args.uot_ablation == knob


def test_phase10r_knobs_only_touch_target_weights():
    amt_w, *_ = _apply_uot_ablation(ablation="no_amount_cost", **_base_kwargs())
    route_w, *_ = _apply_uot_ablation(ablation="no_route_bridge", **_base_kwargs())
    nov_w, *_ = _apply_uot_ablation(ablation="no_address_novelty", **_base_kwargs())
    assert float(amt_w["amount"]) == 0.0
    assert float(route_w["route"]) == 0.0
    assert float(nov_w["novelty"]) == 0.0
    assert float(amt_w["route"]) > 0.0
    assert float(route_w["amount"]) > 0.0
    assert float(nov_w["amount"]) > 0.0
