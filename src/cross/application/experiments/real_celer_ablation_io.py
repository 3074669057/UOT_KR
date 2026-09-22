"""Real Celer Transport Solver Ablation - I/O helpers, dataclasses, and file formats."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _file_content_hash(path, max_bytes=1_000_000):
    """SHA256 of first max_bytes of file content."""
    if not path or not Path(path).is_file():
        return "0" * 12
    with open(path, "rb") as f:
        data = f.read(max_bytes)
    return hashlib.sha256(data).hexdigest()[:12]


@dataclass
class AblationRunConfig:
    """Immutable run configuration for a single solver ablation run."""
    run_id: str
    solver: str
    mode: str
    run_root: Path
    force: bool = False
    base_config_path: Path | None = None
    input_csv: Path | None = None
    label_csv: Path | None = None
    gt_pairs_csv: Path | None = None
    time_delay_policy: str = "tx_if_available_else_flow_representative"
    max_delay_sec: float = 21600.0
    causal_violation_penalty: float = 5.0
    decode_threshold: float = 0.01
    weak_share_threshold: float = 0.12
    apply_joint_time_filter: bool = True
    flow_dst_top_k: int = 200
    flow_max_matrix_cells: int = 12_000_000
    reg: float = 0.05
    reg_m: float = 0.5
    backend: str = "pot"
    cost_weights: dict[str, float] | None = None
    use_graph: bool = True
    use_risk_weighted_source_mass: bool = True
    use_evidence_weighted_target_mass: bool = True
    lambda_risk: float = 0.25
    seed: int = 42
    sample_frac: float = 1.0
    admissible_decode_policy: str = "joint_time_admissible_filter_only"
    topk_rescue_enabled: bool = False
    candidate_pool_source: str = "phase10r"
    allow_dense_debug: bool = False
    allow_nonzero_after_cvr: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["run_root"] = str(self.run_root)
        if self.base_config_path is not None:
            d["base_config_path"] = str(self.base_config_path)
        if self.input_csv is not None:
            d["input_csv"] = str(self.input_csv)
        if self.label_csv is not None:
            d["label_csv"] = str(self.label_csv)
        if self.gt_pairs_csv is not None:
            d["gt_pairs_csv"] = str(self.gt_pairs_csv)
        return d

    @property
    def hash(self) -> str:
        """Config hash covering solver, cost weights, delay policy, thresholds, candidate params, time filter params."""
        payload = {
            "solver": self.solver,
            "cost_weights": self.cost_weights,
            "time_delay_policy": self.time_delay_policy,
            "max_delay_sec": self.max_delay_sec,
            "causal_violation_penalty": self.causal_violation_penalty,
            "decode_threshold": self.decode_threshold,
            "weak_share_threshold": self.weak_share_threshold,
            "apply_joint_time_filter": self.apply_joint_time_filter,
            "flow_dst_top_k": self.flow_dst_top_k,
            "flow_max_matrix_cells": self.flow_max_matrix_cells,
            "reg": self.reg,
            "reg_m": self.reg_m,
            "backend": self.backend,
            "use_graph": self.use_graph,
            "use_risk_weighted_source_mass": self.use_risk_weighted_source_mass,
            "use_evidence_weighted_target_mass": self.use_evidence_weighted_target_mass,
            "lambda_risk": self.lambda_risk,
            "seed": self.seed,
            "admissible_decode_policy": self.admissible_decode_policy,
            "topk_rescue_enabled": self.topk_rescue_enabled,
            "candidate_pool_source": self.candidate_pool_source,
            "flow_dst_top_k": self.flow_dst_top_k,
            "flow_max_matrix_cells": self.flow_max_matrix_cells,
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

@dataclass
class AblationResult:
    """Output bundle for a single solver run."""
    run_id: str
    solver: str
    config_hash: str
    input_hash: str
    skipped: bool = False
    skip_reason: str = ""
    metrics: dict = field(default_factory=dict)
    solver_meta: dict = field(default_factory=dict)
    output_paths: dict = field(default_factory=dict)


@dataclass
class RealCelerInputs:
    """Canonical input bundle shared across all solver runs."""
    eth_df: Any = None
    bnb_df: Any = None
    src_flows: list = field(default_factory=list)
    dst_flows: list = field(default_factory=list)
    gt_tx_pairs: list = field(default_factory=list)
    flow_gt_pairs: Any = None
    covered_quotient_pairs: Any = None
    candidate_pool: Any = None
    src_all: Any = None
    candidate_mask: Any = None
    dst_norm: Any = None
    missing_inputs: list = field(default_factory=list)

    @property
    def n_src_flow(self) -> int:
        return len(self.src_flows)

    @property
    def n_dst_flow(self) -> int:
        return len(self.dst_flows)

    @property
    def n_gt_tx_pairs(self) -> int:
        return len(self.gt_tx_pairs)


def _json_hash(obj):
    raw = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def compute_input_hash(
    src_flows, dst_flows, gt_pairs,
    src_flows_path=None, dst_flows_path=None, label_csv_path=None,
):
    """Input hash covering counts, prefixes, and file content fingerprints."""
    payload = {
        "n_src": len(src_flows),
        "n_dst": len(dst_flows),
        "n_gt_pairs": len(gt_pairs),
        "gt_pairs_prefix": sorted(gt_pairs)[:20],
        "src_flow_ids_prefix": sorted(f.get("flow_id", "") for f in src_flows)[:20],
        "dst_flow_ids_prefix": sorted(f.get("flow_id", "") for f in dst_flows)[:20],
    }
    if src_flows_path:
        payload["src_flows_hash"] = _file_content_hash(Path(src_flows_path))
    if dst_flows_path:
        payload["dst_flows_hash"] = _file_content_hash(Path(dst_flows_path))
    if label_csv_path:
        payload["label_csv_hash"] = _file_content_hash(Path(label_csv_path))
    return _json_hash(payload)


def compute_config_hash(config):
    return config.hash
DEFAULT_COST_WEIGHTS = {
    "amount": 0.35, "time": 0.25, "route": 0.15,
    "risk": 0.15, "graph": 0.05, "evidence": 0.05, "novelty": 0.05,
}

SOLVER_NAMES = ["rc_uot_full","rc_uot_numpy","balanced_sinkhorn","balanced_emd","hungarian","greedy_nn","cost_ranking"]

SOLVER_MINIMAL_DEFAULT = ["rc_uot_full","cost_ranking","hungarian","greedy_nn","balanced_sinkhorn"]

SOLVER_FULL_DEFAULT = ["rc_uot_full","rc_uot_numpy","cost_ranking","hungarian","greedy_nn","balanced_sinkhorn","balanced_emd"]


def build_solver_runs(run_root, mode, solvers=None, **kwargs):
    """Generate AblationRunConfig list from mode + solver selection."""
    if mode == "solver_minimal":
        sel = solvers or SOLVER_MINIMAL_DEFAULT
    elif mode == "solver_full":
        sel = solvers or SOLVER_FULL_DEFAULT
    elif mode == "cost_components":
        sel = solvers or ["rc_uot_full"]
    elif mode == "all_registered":
        sel = solvers or SOLVER_NAMES
    else:
        sel = solvers or SOLVER_MINIMAL_DEFAULT

    runs = []
    for sn in sel:
        if sn not in SOLVER_NAMES:
            print(f"[WARN] Unknown solver {sn!r}, skipping")
            continue
        run_id = f"real_celer_solver__{sn}"
        backend = "numpy" if sn == "rc_uot_numpy" else "pot"
        cfg = AblationRunConfig(run_id=run_id, solver=sn, mode=mode, run_root=run_root, backend=backend, **kwargs)
        runs.append(cfg)
    return runs

def run_output_dir(run_root, run_id):
    return Path(run_root) / "runs" / run_id


def write_run_artifacts(result, config, config_dict, meta, transport_plan_rows, topk_rows, *, run_dir=None):
    """Persist all per-run output files. Returns path map."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    (run_dir/"config.json").write_text(json.dumps(config_dict,indent=2,default=str),encoding="utf-8")
    paths["config_json"] = str(run_dir/"config.json")
    (run_dir/"meta.json").write_text(json.dumps(meta,indent=2,default=str),encoding="utf-8")
    paths["meta_json"] = str(run_dir/"meta.json")
    (run_dir/"metrics.json").write_text(json.dumps(result.metrics,indent=2,default=str),encoding="utf-8")
    paths["metrics_json"] = str(run_dir/"metrics.json")
    (run_dir/"solver_meta.json").write_text(json.dumps(result.solver_meta,indent=2,default=str),encoding="utf-8")
    paths["solver_meta_json"] = str(run_dir/"solver_meta.json")
    tpp = run_dir / "transport_plan.csv"
    (pd.DataFrame(transport_plan_rows) if transport_plan_rows else pd.DataFrame()).to_csv(tpp,index=False)
    paths["transport_plan_csv"] = str(tpp)
    tkc = run_dir / "topk_candidates.csv"
    (pd.DataFrame(topk_rows) if topk_rows else pd.DataFrame()).to_csv(tkc,index=False)
    paths["topk_candidates_csv"] = str(tkc)
    return paths


def read_metrics_json(run_dir):
    mp = Path(run_dir) / "metrics.json"
    return json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else None


def build_meta(config, config_hash, input_hash, inputs, cost_context_meta):
    import platform
    from datetime import datetime, timezone
    return {
        "run_id": config.run_id,
        "solver": config.solver,
        "mode": config.mode,
        "apply_joint_time_filter": config.apply_joint_time_filter,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
        "input_hash": input_hash,
        "n_src_flow": inputs.n_src_flow,
        "n_dst_flow": inputs.n_dst_flow,
        "n_gt_tx_pairs": inputs.n_gt_tx_pairs,
        "missing_inputs": inputs.missing_inputs,
        "cost_context": cost_context_meta,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": config.seed,
    }
