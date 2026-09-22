from __future__ import annotations

import numpy as np

from cross.domain.uot.cost_matrix import build_cost_matrix, build_cost_matrix_decomposed


def test_cost_matrix_shape_and_range():
    s = [
        {
            "flow_id": "e1",
            "amount_usd": 100.0,
            "start_time": 0.0,
            "end_time": 100.0,
            "aml_score": 0.5,
            "bridge": "celer",
            "graph_embedding": [1.0, 0.0, 0.0],
        }
    ]
    t = [
        {
            "flow_id": "b1",
            "amount_usd": 100.0,
            "start_time": 200.0,
            "end_time": 200.0,
            "aml_score": 0.5,
            "bridge": "celer",
            "graph_embedding": [1.0, 0.0, 0.0],
        }
    ]
    c = build_cost_matrix(s, t, use_graph=True)
    assert c.shape == (1, 1)
    assert 0.0 <= float(c[0, 0]) <= 1.0


def test_build_cost_matrix_decomposed_keys():
    s = [{"flow_id": "e1", "amount_usd": 1.0, "start_time": 0.0, "end_time": 1.0, "aml_score": 0.0, "bridge": "x", "graph_embedding": None}]
    t = [{"flow_id": "b1", "amount_usd": 1.0, "start_time": 2.0, "end_time": 2.0, "aml_score": 0.0, "bridge": "x", "graph_embedding": None}]
    d = build_cost_matrix_decomposed(s, t, use_graph=False)
    assert d["C"].shape == (1, 1)
    assert d["amount_cost"].shape == (1, 1)
    assert d["time_cost"].shape == (1, 1)


def test_legacy_vs_representative_delay_differs():
    s = [{"flow_id": "e1", "amount_usd": 1.0, "start_time": 0.0, "end_time": 100.0, "aml_score": 0.0, "bridge": "x", "graph_embedding": None}]
    t = [{"flow_id": "b1", "amount_usd": 1.0, "start_time": 200.0, "end_time": 300.0, "aml_score": 0.0, "bridge": "x", "graph_embedding": None}]
    d_legacy = build_cost_matrix_decomposed(s, t, use_graph=False, time_delay_policy="legacy_flow_boundary")
    d_repr = build_cost_matrix_decomposed(s, t, use_graph=False, time_delay_policy="tx_if_available_else_flow_representative")
    assert float(d_legacy["delay_sec"][0, 0]) == 100.0  # dst.start - src.end
    assert float(d_repr["delay_sec"][0, 0]) == 300.0  # dst.end - src.start
    assert float(d_legacy["delay_sec"][0, 0]) != float(d_repr["delay_sec"][0, 0])


def test_time_causal_violation_is_max_cost():
    s = [{"flow_id": "e1", "amount_usd": 1.0, "start_time": 500.0, "end_time": 500.0, "aml_score": 0.0, "bridge": "x", "graph_embedding": None}]
    t = [{"flow_id": "b1", "amount_usd": 1.0, "start_time": 100.0, "end_time": 100.0, "aml_score": 0.0, "bridge": "x", "graph_embedding": None}]
    c = build_cost_matrix(s, t, use_graph=False)
    assert float(c[0, 0]) >= 0.24
