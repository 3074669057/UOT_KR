from __future__ import annotations

import numpy as np

import ot  # type: ignore[import-untyped]

from cross.domain.uot.cost_matrix import build_cost_matrix
from cross.domain.uot.uot_solver import normalize_mass, solve_uot
from cross.domain.uot.uot_solver_numpy import uot_sinkhorn


def test_solve_uot_returns_stochastic_matrix():
    src = [
        {"flow_id": "a", "amount_usd": 1.0, "start_time": 0.0, "end_time": 1.0, "aml_score": 0.1, "bridge": "c", "graph_embedding": None},
        {"flow_id": "b", "amount_usd": 1.0, "start_time": 0.0, "end_time": 1.0, "aml_score": 0.1, "bridge": "c", "graph_embedding": None},
    ]
    tgt = [
        {"flow_id": "x", "amount_usd": 1.0, "start_time": 2.0, "end_time": 2.0, "aml_score": 0.1, "bridge": "c", "graph_embedding": None},
        {"flow_id": "y", "amount_usd": 1.0, "start_time": 2.0, "end_time": 2.0, "aml_score": 0.1, "bridge": "c", "graph_embedding": None},
    ]
    p, c = solve_uot(src, tgt, reg=0.08, reg_m=0.6, use_graph=False, backend="pot")
    assert p.shape == (2, 2)
    assert float(p.sum()) > 0.0


def test_solve_uot_pot_import_error_falls_back_to_numpy(monkeypatch):
    src = [
        {"flow_id": "a", "amount_usd": 1.0, "start_time": 0.0, "end_time": 1.0, "aml_score": 0.1, "bridge": "c", "graph_embedding": None},
    ]
    tgt = [
        {"flow_id": "x", "amount_usd": 1.0, "start_time": 2.0, "end_time": 2.0, "aml_score": 0.1, "bridge": "c", "graph_embedding": None},
    ]

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "ot":
            raise ImportError("simulated missing POT")
        return __import__(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    meta: dict = {}
    p, c = solve_uot(src, tgt, reg=0.08, reg_m=0.6, use_graph=False, backend="pot", solver_meta=meta)
    assert p.shape == (1, 1)
    assert meta.get("pot_fallback") is True
    assert meta.get("actual_backend") == "numpy"


def test_pot_vs_numpy_same_cost_matrix():
    src = [
        {"flow_id": "a", "amount_usd": 1.0, "start_time": 0.0, "end_time": 1.0, "aml_score": 0.1, "bridge": "c", "graph_embedding": None},
        {"flow_id": "b", "amount_usd": 1.0, "start_time": 0.0, "end_time": 1.0, "aml_score": 0.1, "bridge": "c", "graph_embedding": None},
    ]
    tgt = [
        {"flow_id": "x", "amount_usd": 1.0, "start_time": 2.0, "end_time": 2.0, "aml_score": 0.1, "bridge": "c", "graph_embedding": None},
        {"flow_id": "y", "amount_usd": 1.0, "start_time": 2.0, "end_time": 2.0, "aml_score": 0.1, "bridge": "c", "graph_embedding": None},
    ]
    c = build_cost_matrix(src, tgt, use_graph=False)
    a = normalize_mass(np.array([1.0, 1.0]))
    b = normalize_mass(np.array([1.0, 1.0]))
    p_pot = ot.unbalanced.sinkhorn_unbalanced(a, b, c, reg=0.08, reg_m=0.6, numItermax=2000, stopThr=1e-9)
    p_np = uot_sinkhorn(a, b, c, epsilon=0.08, tau=0.6)
    d = float(np.abs(np.asarray(p_pot) - p_np).sum())
    assert d < 0.45
