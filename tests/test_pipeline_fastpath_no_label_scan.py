"""Regression: fast-path CLI modes must not run ``label_file_usable`` before their branch."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cross.application.pipeline import run_pipeline
from cross.interfaces.cli import build_parser


def _minimal_cfg() -> dict:
    return {"uot": {"reg": 0.05}}


def _fake_for(target: str):
    if "decode_threshold_sweep" in target:

        def fn(out, **kw):
            outp = Path(out)
            (outp / "experiments").mkdir(parents=True, exist_ok=True)
            return outp / "experiments" / "decode_threshold_sweep.csv"

        return fn
    if "run_candidate_recall_diagnostics" in target:

        def fn(*a, **k):
            return {}

        return fn
    if "run_candidate_pool_sweep" in target:

        def fn(*a, **k):
            outp = Path(a[0])
            (outp / "experiments").mkdir(parents=True, exist_ok=True)
            return outp / "experiments" / "candidate_pool_sweep.csv"

        return fn
    if "write_synthetic_failure_debug_csvs" in target:

        def fn(*a, **k):
            return Path("a.csv"), Path("b.csv")

        return fn
    if "run_paper_finalization" in target:

        def fn(*a, **k):
            return 0

        return fn

    def fn(*a, **k):
        return 0

    return fn


@pytest.mark.parametrize(
    "extra_cli,patch_target",
    [
        (["--paper-experiment-closure"], "cross.application.paper_experiment_closure.run_paper_experiment_closure"),
        (["--flow-uot-resume"], "cross.application.paper_experiment_closure.run_flow_uot_resume"),
        (["--candidate-recall-diagnostics-fast"], "cross.domain.uot.candidate_recall_diagnostics.run_candidate_recall_diagnostics"),
        (["--decode-threshold-sweep"], "cross.domain.evaluation.decode_threshold_sweep.run_decode_threshold_sweep"),
        (["--synthetic-failure-debug"], "cross.domain.evaluation.synthetic_failure_debug.write_synthetic_failure_debug_csvs"),
        (["--candidate-pool-sweep-only", "G_larger_budget_18M"], "cross.domain.uot.candidate_recall_diagnostics.run_candidate_pool_sweep"),
        (["--paper-finalization"], "cross.experiments.paper_finalize_bundle.run_paper_finalization"),
    ],
)
def test_fastpath_never_calls_label_file_usable(tmp_path, monkeypatch, extra_cli, patch_target):
    cfg_path = tmp_path / "defaults.json"
    loc_path = tmp_path / "local.json"
    cfg_path.write_text(json.dumps({"uot": {}, "output_dir": "out"}), encoding="utf-8")
    loc_path.write_text("{}", encoding="utf-8")

    outdir = tmp_path / "runout"
    outdir.mkdir(parents=True, exist_ok=True)

    parser = build_parser()
    argv = [*extra_cli, "--out", str(outdir), "--config", str(cfg_path), "--local-config", str(loc_path)]
    args = parser.parse_args(argv)

    called: list[str] = []

    def boom(label):
        called.append(str(label))
        return True

    monkeypatch.setattr("cross.application.pipeline.label_file_usable", boom)
    monkeypatch.setattr(patch_target, _fake_for(patch_target))

    rc = run_pipeline(args, cfg=_minimal_cfg())
    assert rc == 0
    assert called == []
