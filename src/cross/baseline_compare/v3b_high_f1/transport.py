"""Transport matrix cache for v3b (read-only v3 / production)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cross.application.experiments.uot_cache_utils import (
    build_cost_matrix_from_components,
    load_cost_component_cache,
    load_flow_segments,
    solve_from_cache,
)
from cross.baseline_compare.rc_uot_q_pilot import prepare_rc_flows_and_weights
from cross.baseline_compare.v3b_high_f1.constants import OUT_V3, UOT_BASE
from cross.baseline_compare.v3b_high_f1.masks import (
    CHAIN_OBSERVABLES_MASK,
    get_v3b_mask_spec,
    prepare_chain_observables_flows,
)
from cross.domain.uot.delay_policy import DEFAULT_TIME_DELAY_POLICY


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def transport_cache_path(output_dir: Path, mask_id: str) -> Path:
    return output_dir / "transport_cache" / mask_id / "transport_matrix.npz"


def load_or_compute_transport(
    *,
    mask_id: str,
    output_dir: Path,
    uot_base: Path = UOT_BASE,
) -> tuple[np.ndarray, list, list, dict[str, float], dict[str, Any]]:
    cache_path = transport_cache_path(output_dir, mask_id)
    meta_path = cache_path.parent / "transport_meta.json"

    if cache_path.is_file() and meta_path.is_file():
        p = np.asarray(np.load(cache_path)["P"], dtype=float)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        eth_f = load_flow_segments(uot_base / "uot" / "uot_flow_segments_eth.csv")
        bnb_f = load_flow_segments(uot_base / "uot" / "uot_flow_segments_bnb.csv")
        spec = get_v3b_mask_spec(mask_id)
        if mask_id == CHAIN_OBSERVABLES_MASK:
            eth_f, bnb_f, weights, prep = prepare_chain_observables_flows(eth_f, bnb_f)
        else:
            eth_f, bnb_f, weights, prep = prepare_rc_flows_and_weights(spec, eth_f, bnb_f)
        return p, eth_f, bnb_f, weights, {**meta, **prep}

    v3_npz = OUT_V3 / "masks" / mask_id / "rc_uot_q" / "transport_matrix.npz"
    if v3_npz.is_file() and mask_id != CHAIN_OBSERVABLES_MASK:
        p = np.asarray(np.load(v3_npz)["P"], dtype=float)
        eth = load_flow_segments(uot_base / "uot" / "uot_flow_segments_eth.csv")
        bnb = load_flow_segments(uot_base / "uot" / "uot_flow_segments_bnb.csv")
        spec = get_v3b_mask_spec(mask_id)
        eth_f, bnb_f, weights, prep = prepare_rc_flows_and_weights(spec, eth, bnb)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, P=p)
        _write_json(meta_path, {"source": "copied_from_v3_readonly", "v3_path": str(v3_npz)})
        return p, eth_f, bnb_f, weights, prep

    cache = load_cost_component_cache(uot_base)
    eth = load_flow_segments(uot_base / "uot" / "uot_flow_segments_eth.csv")
    bnb = load_flow_segments(uot_base / "uot" / "uot_flow_segments_bnb.csv")
    spec = get_v3b_mask_spec(mask_id)
    if mask_id == CHAIN_OBSERVABLES_MASK:
        eth_f, bnb_f, weights, prep = prepare_chain_observables_flows(eth, bnb)
    else:
        eth_f, bnb_f, weights, prep = prepare_rc_flows_and_weights(spec, eth, bnb)

    c_mat = build_cost_matrix_from_components(
        cache["components"],
        time_weight=1.0,
        causal_weight=1.0,
        baseline_causal_penalty=float(cache["baseline_causal_penalty"]),
        max_delay_sec=float(cache["max_delay_sec"]),
        base_weights=weights,
        delay_policy=DEFAULT_TIME_DELAY_POLICY,
        source_flows=eth_f,
        target_flows=bnb_f,
    )
    p = solve_from_cache(cache, c_mat)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, P=p)
    _write_json(meta_path, {"source": "computed_v3b_cache", "mask_id": mask_id})
    return p, eth_f, bnb_f, weights, prep
