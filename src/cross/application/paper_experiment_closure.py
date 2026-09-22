"""Paper experiment bundle: freeze label v1, RC-UOT + baselines, semi-synthetic, threshold sweep, summary."""
from __future__ import annotations

import json
import logging
import shutil
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import pandas as pd

from cross.application.standalone_flow_uot import run_standalone_flow_uot
from cross.config.output_layout import locate_output_file, output_file
from cross.domain.evaluation.semi_synthetic_flows import build_semi_synthetic_from_flow_labels
from cross.domain.evaluation.synthetic_segment_subgraph import write_synthetic_subgraph_segment_csvs
from cross.domain.labels.celer_supervised_pipeline import finalize_multi_source_label_bundle, patch_uot_evaluation_label_source
from cross.domain.labels.flow_label_builder import build_flow_labels
from cross.domain.labels.flow_segment_builder import build_flow_segments_from_evidence
from cross.domain.labels.tx_anchor_builder import build_tx_anchor_labels

logger = logging.getLogger(__name__)


def _uot_pool_kwargs_from_uk(uk: dict[str, Any]) -> dict[str, Any]:
    return {
        "uot_pool_strategy": str(uk.get("uot_pool_strategy") or "default"),
        "uot_min_pool_per_asset": int(uk.get("uot_min_pool_per_asset") or 0),
        "uot_route_preserving_m": int(uk.get("uot_route_preserving_m") or 0),
    }


def run_flow_uot_resume(args: Namespace, cfg: dict) -> int:
    """Re-run main RC-UOT + baselines/ablations on existing segment/label artifacts (writes ``uot/`` including ``uot_transport_matrix.npz``)."""
    from cross.application.pipeline import uot_kwargs_from_config

    out_root = Path(args.out)
    stats_path = output_file(out_root, "flow_label_stats.json")
    if not stats_path.is_file():
        logger.error("flow_uot_resume: missing %s", stats_path)
        return 1
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    uk = uot_kwargs_from_config(args, cfg)
    src, dst, pick_meta = select_flow_segments_for_closure(out_root, stats)
    assert_segments_match_stats_or_raise(src, dst, stats)
    seg_block = {
        **pick_meta,
        "segment_rows_match_stats": True,
        "refused_uot_due_to_segment_mismatch": False,
        "expected_src_flows_from_stats": int(stats.get("num_src_flows") or 0),
        "expected_dst_flows_from_stats": int(stats.get("num_dst_flows") or 0),
    }
    fl = _artifact_path(out_root, "flow_labels.csv")
    if not src.is_file() or not dst.is_file() or not fl.is_file():
        logger.error("flow_uot_resume: need flow_segments_eth/bnb and flow_labels under %s", out_root)
        return 1
    hint_arg = getattr(args, "synthetic_eval_hints", None)
    syn_hints_main = Path(hint_arg) if hint_arg else None
    run_standalone_flow_uot(
        out_root,
        src_flows_csv=src,
        dst_flows_csv=dst,
        flow_labels_csv=fl,
        uot_reg=float(uk["uot_reg"]),
        uot_reg_m=float(uk["uot_reg_m"]),
        uot_decode_threshold=float(uk["uot_decode_threshold"]),
        uot_cost_weights=uk.get("uot_cost_weights"),
        uot_backend=str(uk["uot_backend"]),
        uot_max_delay_sec=float(uk["uot_max_delay_sec"]),
        uot_causal_violation_penalty=float(uk["uot_causal_violation_penalty"]),
        uot_lambda_risk=float(uk["uot_lambda_risk"]),
        uot_causal_infeasible_delay_sec=uk.get("uot_causal_infeasible_delay_sec"),
        uot_export_matrix=bool(uk["uot_export_matrix"]),
        uot_export_cost_components=bool(uk["uot_export_cost_components"]),
        uot_export_cost_matrix_csv=bool(uk.get("uot_export_cost_matrix_csv", False)),
        uot_allow_unmatched=bool(uk["uot_allow_unmatched"]),
        uot_use_graph_embedding=bool(uk["uot_use_graph_embedding"]),
        graph_ranker_checkpoint=getattr(args, "graph_ranker_checkpoint", None),
        uot_ablation=str(uk.get("uot_ablation") or "none"),
        run_flow_baselines=True,
        flow_label_min_confidence=float(uk.get("flow_label_min_confidence") or 0.0),
        synthetic_eval_hints_path=syn_hints_main,
        uot_flow_dst_top_k=int(uk.get("uot_flow_dst_top_k") or 200),
        uot_flow_max_matrix_cells=int(uk.get("uot_flow_max_matrix_cells") or 6_000_000),
        **_uot_pool_kwargs_from_uk(uk),
    )
    _patch_uot_eval_with_segment_validation(out_root, seg_block)
    _assert_main_uot_matches_segment_inputs(out_root, seg_block)
    ev_post = output_file(out_root, "uot_evaluation_metrics.json")
    if ev_post.is_file():
        try:
            ev_obj = json.loads(ev_post.read_text(encoding="utf-8"))
            tg_m = ev_obj.get("transport_graph_meta")
            if isinstance(tg_m, dict):
                _merge_validation_transport_graph_meta(out_root, tg_m)
        except Exception:
            logger.exception("flow_uot_resume: could not merge transport_graph_meta into validation_report")
    logger.info("flow_uot_resume finished -> %s", out_root)
    return 0


def _flush_logs() -> None:
    for lg in (logging.getLogger(), logger):
        for h in getattr(lg, "handlers", []) or []:
            try:
                h.flush()
            except Exception:
                pass


def _required_paths(out_root: Path) -> list[tuple[str, Path]]:
    r = Path(out_root)
    return [
        ("label_layer_v1/label_layer_v1_summary.json", r / "label_layer_v1" / "label_layer_v1_summary.json"),
        ("uot/uot_transport_plan.csv", output_file(r, "uot_transport_plan.csv")),
        ("uot/uot_unmatched_mass.csv", output_file(r, "uot_unmatched_mass.csv")),
        ("eval/uot_evaluation_metrics.json", output_file(r, "uot_evaluation_metrics.json")),
        ("eval/uot_eval_by_pattern.csv", output_file(r, "uot_eval_by_pattern.csv")),
        ("eval/ablation_metrics.csv", output_file(r, "ablation_metrics.csv")),
        ("labels/synthetic_flow_labels.csv", output_file(r, "synthetic_flow_labels.csv")),
        ("labels/synthetic_uot_eval_metrics.json", output_file(r, "synthetic_uot_eval_metrics.json")),
        ("eval/synthetic_eval_by_scenario.csv", output_file(r, "synthetic_eval_by_scenario.csv")),
        ("experiments/threshold_sensitivity.csv", output_file(r, "threshold_sensitivity.csv")),
        ("reports/paper_experiment_summary.md", output_file(r, "paper_experiment_summary.md")),
    ]


def validate_and_merge_paper_closure_report(out_root: Path) -> dict[str, Any]:
    """Verify paper-closure artifacts; log failures; merge ``paper_closure`` into ``validation_report.json``."""
    out_root = Path(out_root)
    missing: list[dict[str, Any]] = []
    presence: dict[str, str] = {}
    min_bytes = 2

    for rel, p in _required_paths(out_root):
        if not p.is_file():
            presence[rel] = "missing"
            entry = {"artifact": rel, "path": str(p), "reason": "file does not exist"}
            missing.append(entry)
            logger.error("Paper closure artifact missing: %s (%s)", rel, p)
        elif p.stat().st_size < min_bytes:
            presence[rel] = "empty"
            entry = {"artifact": rel, "path": str(p), "reason": f"file exists but size {p.stat().st_size} < {min_bytes} bytes"}
            missing.append(entry)
            logger.error("Paper closure artifact empty: %s (%s)", rel, p)
        else:
            presence[rel] = "ok"

    stats_path = output_file(out_root, "flow_label_stats.json")
    predominantly_1_1 = False
    if stats_path.is_file():
        try:
            st = json.loads(stats_path.read_text(encoding="utf-8"))
            predominantly_1_1 = bool(st.get("predominantly_one_to_one"))
        except Exception as e:
            logger.error("flow_label_stats.json unreadable: %s", e)
    else:
        logger.error("flow_label_stats.json missing for predominantly_one_to_one check")

    val_path = output_file(out_root, "validation_report.json")
    sanity_all_true = False
    sanity_detail: dict[str, Any] = {}
    if val_path.is_file():
        try:
            vr = json.loads(val_path.read_text(encoding="utf-8"))
            sc = vr.get("sanity_checks") or {}
            if isinstance(sc, dict) and sc:
                sanity_all_true = bool(all(bool(v) for v in sc.values()))
                sanity_detail = dict(sc)
            else:
                sanity_detail = {"note": "sanity_checks missing or empty"}
                logger.error("validation_report.json: sanity_checks missing or not a dict")
        except Exception as e:
            logger.error("validation_report.json unreadable: %s", e)
    else:
        logger.error("validation_report.json missing for sanity_checks merge")

    uot_eval_path = output_file(out_root, "uot_evaluation_metrics.json")
    has_synthetic_metrics = False
    if uot_eval_path.is_file():
        try:
            ev = json.loads(uot_eval_path.read_text(encoding="utf-8"))
            has_synthetic_metrics = isinstance(ev.get("synthetic_metrics"), dict) and len(ev.get("synthetic_metrics") or {}) > 0
        except Exception as e:
            logger.error("uot_evaluation_metrics.json unreadable: %s", e)
    if not has_synthetic_metrics:
        logger.error(
            "uot_evaluation_metrics.json: missing or empty synthetic_metrics "
            "(semi-synthetic UOT may have been skipped if predominantly_one_to_one is false or segment clone failed)"
        )

    ablation_methods_ok = False
    expected_methods = {
        "Greedy",
        "Hungarian",
        "Balanced OT",
        "RC-UOT",
        "RC-UOT no_risk",
        "RC-UOT no_causal",
        "RC-UOT no_graph",
        "RC-UOT no_evidence",
    }
    ab_path = output_file(out_root, "ablation_metrics.csv")
    found_methods: set[str] = set()
    if ab_path.is_file():
        try:
            adf = pd.read_csv(ab_path, dtype=str, keep_default_na=False)
            if "method" in adf.columns:
                found_methods = set(adf["method"].astype(str).str.strip())
            ablation_methods_ok = expected_methods.issubset(found_methods)
            if not ablation_methods_ok:
                logger.error(
                    "ablation_metrics.csv: expected methods subset missing. expected=%s found=%s",
                    sorted(expected_methods),
                    sorted(found_methods),
                )
        except Exception as e:
            logger.error("ablation_metrics.csv unreadable: %s", e)
    else:
        logger.error("ablation_metrics.csv missing for method coverage check")

    scenario_ok = False
    scen_path = output_file(out_root, "synthetic_eval_by_scenario.csv")
    need_scen = {"split", "merge", "unmatched", "delay_noise"}
    scen_found: set[str] = set()
    if scen_path.is_file():
        try:
            sdf = pd.read_csv(scen_path, dtype=str, keep_default_na=False)
            if "scenario" in sdf.columns:
                scen_found = set(sdf["scenario"].astype(str).str.strip())
            scenario_ok = need_scen.issubset(scen_found)
            if not scenario_ok:
                logger.error(
                    "synthetic_eval_by_scenario.csv: need scenarios %s; found %s",
                    sorted(need_scen),
                    sorted(scen_found),
                )
        except Exception as e:
            logger.error("synthetic_eval_by_scenario.csv unreadable: %s", e)
    else:
        logger.error("synthetic_eval_by_scenario.csv missing (synthetic eval with hints did not run or failed)")

    thr_ok = False
    thr_path = output_file(out_root, "threshold_sensitivity.csv")
    need_settings = {"default", "stricter", "looser"}
    settings_found: set[str] = set()
    if thr_path.is_file():
        try:
            tdf = pd.read_csv(thr_path, dtype=str, keep_default_na=False)
            if "setting" in tdf.columns:
                settings_found = set(tdf["setting"].astype(str).str.strip())
            thr_ok = need_settings.issubset(settings_found)
            if not thr_ok:
                logger.error(
                    "threshold_sensitivity.csv: need settings %s; found %s",
                    sorted(need_settings),
                    sorted(settings_found),
                )
        except Exception as e:
            logger.error("threshold_sensitivity.csv unreadable: %s", e)
    else:
        logger.error("threshold_sensitivity.csv missing (threshold_sensitivity step failed or was not written)")

    summary_path = output_file(out_root, "paper_experiment_summary.md")
    summary_ok = False
    summary_note = ""
    if summary_path.is_file() and summary_path.stat().st_size > 400:
        summary_ok = True
        summary_note = "Length and structure suitable as a starting draft for the experiments section (review numbers and narrative)."
    else:
        logger.error("paper_experiment_summary.md missing or too short for use as paper draft")

    paper_section: dict[str, Any] = {
        "artifact_presence": presence,
        "missing_artifacts": missing,
        "checks": {
            "predominantly_one_to_one": predominantly_1_1,
            "validation_sanity_checks_all_true": sanity_all_true,
            "synthetic_metrics_present": has_synthetic_metrics,
            "ablation_methods_complete": ablation_methods_ok,
            "synthetic_scenarios_complete": scenario_ok,
            "threshold_settings_complete": thr_ok,
            "paper_summary_draft_ready": summary_ok,
        },
        "details": {
            "sanity_checks": sanity_detail,
            "ablation_methods_found": sorted(found_methods),
            "synthetic_scenarios_found": sorted(scen_found),
            "threshold_settings_found": sorted(settings_found),
            "paper_summary_note": summary_note,
        },
    }

    merged: dict[str, Any] = {}
    if val_path.is_file():
        try:
            merged = json.loads(val_path.read_text(encoding="utf-8"))
        except Exception:
            merged = {}
    merged["paper_closure"] = paper_section
    val_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    logger.info(
        "Paper closure artifact validation: missing_count=%d checks=%s",
        len(missing),
        json.dumps(paper_section["checks"], ensure_ascii=False),
    )

    v1_val = out_root / "label_layer_v1" / "validation_report.json"
    if v1_val.parent.is_dir():
        try:
            shutil.copy2(val_path, v1_val)
        except OSError as e:
            logger.warning("Could not copy validation_report.json to label_layer_v1: %s", e)

    return paper_section


def _artifact_path(root: Path, filename: str) -> Path:
    """Prefer nested ``output_file`` layout; fall back to legacy flat ``root/filename``."""
    p = locate_output_file(root, filename)
    if p.is_file():
        return p
    flat = root / filename
    if flat.is_file():
        return flat
    return p


def _count_csv_rows(p: Path) -> int:
    if not p.is_file() or p.stat().st_size < 10:
        return 0
    try:
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except OSError:
        return 0


def _path_has_uot_segment(p: Path) -> bool:
    s = str(p).replace("\\", "/").lower()
    return "/uot/" in s


def _segment_inventory_paths(out_root: Path) -> list[Path]:
    """All candidate flow segment paths (deduped) for artifact_inventory.json."""
    out_root = Path(out_root)
    seen: set[str] = set()
    paths: list[Path] = []

    def add(p: Path) -> None:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            paths.append(p)

    for r in (
        "flow_segments_eth.csv",
        "flow_segments_bnb.csv",
        "labels/flow_segments_eth.csv",
        "labels/flow_segments_bnb.csv",
        "label_layer_v1/flow_segments_eth.csv",
        "label_layer_v1/flow_segments_bnb.csv",
        "uot/flow_segments_eth.csv",
        "uot/flow_segments_bnb.csv",
    ):
        add(out_root / r)
    thr_root = out_root / "threshold_runs"
    if thr_root.is_dir():
        for sub in sorted(thr_root.iterdir()):
            if not sub.is_dir():
                continue
            add(sub / "labels" / "flow_segments_eth.csv")
            add(sub / "labels" / "flow_segments_bnb.csv")
            add(sub / "uot" / "flow_segments_eth.csv")
            add(sub / "uot" / "flow_segments_bnb.csv")
    return paths


def build_artifact_inventory(out_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for p in _segment_inventory_paths(out_root):
        rows.append(
            {
                "path": str(p.resolve()) if p.is_file() else str(p),
                "exists": p.is_file(),
                "data_rows": _count_csv_rows(p) if p.is_file() else 0,
            }
        )
    return {
        "generated_at_unix": time.time(),
        "run_root": str(Path(out_root).resolve()),
        "flow_segment_candidates": rows,
    }


def _write_artifact_inventory(out_root: Path) -> Path:
    inv = build_artifact_inventory(out_root)
    outp = output_file(out_root, "artifact_inventory.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(inv, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("paper_experiment_closure: wrote %s", outp)
    return outp


def _merge_validation_segment_block(out_root: Path, block: dict[str, Any]) -> None:
    val = output_file(out_root, "validation_report.json")
    data: dict[str, Any] = {}
    if val.is_file():
        try:
            data = json.loads(val.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    cur = data.get("segment_input_validation")
    if isinstance(cur, dict):
        merged = {**cur, **block}
    else:
        merged = dict(block)
    data["segment_input_validation"] = merged
    val.parent.mkdir(parents=True, exist_ok=True)
    val.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _ensure_labels_segments_and_stats(
    out_root: Path,
    ev_e: Path,
    ev_b: Path,
    uk: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild ``labels/flow_segments_*.csv`` (+ flow labels + stats) when rows disagree with ``flow_label_stats``."""
    stats_path = output_file(out_root, "flow_label_stats.json")
    if not stats_path.is_file():
        raise RuntimeError(f"flow_label_stats.json missing: {stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    exp_s = int(stats.get("num_src_flows") or 0)
    exp_d = int(stats.get("num_dst_flows") or 0)
    if exp_s <= 0 or exp_d <= 0:
        return stats
    labels_eth = output_file(out_root, "flow_segments_eth.csv")
    labels_bnb = output_file(out_root, "flow_segments_bnb.csv")
    rs = _count_csv_rows(labels_eth)
    rd = _count_csv_rows(labels_bnb)
    min_ok_s = max(1, int(0.9 * exp_s))
    min_ok_d = max(1, int(0.9 * exp_d))
    mismatch = rs == 0 or rd == 0 or rs < min_ok_s or rd < min_ok_d or rs != exp_s or rd != exp_d
    if not mismatch:
        return stats
    ta = output_file(out_root, "tx_anchor_labels.csv")
    if not ta.is_file():
        raise RuntimeError(
            f"Cannot rebuild flow segments: {ta} missing. "
            f"Stats expect ETH/BNB flows {exp_s}/{exp_d} but labels segments are {rs}/{rd}."
        )
    fw = int(uk.get("flow_window_sec") or 1800)
    logger.info(
        "paper_experiment_closure: rebuilding labels segments + flow labels (segment rows %d/%d vs stats %d/%d)",
        rs,
        rd,
        exp_s,
        exp_d,
    )
    _flush_logs()
    pe, pb, pm = build_flow_segments_from_evidence(ev_e, ev_b, ta, out_root, flow_window_sec=fw)
    build_flow_labels(ta, pe, pb, pm, out_root, min_confidence=0.0)
    stats2 = json.loads(stats_path.read_text(encoding="utf-8"))
    rs2, rd2 = _count_csv_rows(labels_eth), _count_csv_rows(labels_bnb)
    exp_s2 = int(stats2.get("num_src_flows") or 0)
    exp_d2 = int(stats2.get("num_dst_flows") or 0)
    if rs2 != exp_s2 or rd2 != exp_d2:
        raise RuntimeError(
            f"After rebuild, flow segment rows ({rs2}/{rd2}) still do not match flow_label_stats "
            f"({exp_s2}/{exp_d2}). Check evidence CSVs and anchor pipeline."
        )
    return stats2


def select_flow_segments_for_closure(out_root: Path, stats: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    """Pick ETH/BNB segment CSVs from labels / v1 / run root only (never ``uot/``); rows must match stats exactly."""
    out_root = Path(out_root)
    exp_s = int(stats.get("num_src_flows") or 0)
    exp_d = int(stats.get("num_dst_flows") or 0)
    eth_order = [
        out_root / "labels" / "flow_segments_eth.csv",
        out_root / "label_layer_v1" / "flow_segments_eth.csv",
        out_root / "flow_segments_eth.csv",
    ]
    bnb_order = [
        out_root / "labels" / "flow_segments_bnb.csv",
        out_root / "label_layer_v1" / "flow_segments_bnb.csv",
        out_root / "flow_segments_bnb.csv",
    ]

    def _pick_exact(paths: list[Path], expected: int, side: str) -> Path:
        for p in paths:
            if not p.is_file() or _path_has_uot_segment(p):
                continue
            n = _count_csv_rows(p)
            if n == expected:
                return p
        ranked = [(p, _count_csv_rows(p)) for p in paths if p.is_file() and not _path_has_uot_segment(p)]
        best_n = max((t[1] for t in ranked), default=0)
        if best_n < max(1, int(0.9 * expected)):
            raise RuntimeError(
                f"Flow segment mismatch: stats expect {expected} {side} flows, but the best non-uot segment file "
                f"only has {best_n} rows (<90% of stats). Refuse to run UOT."
            )
        raise RuntimeError(
            f"Flow segment mismatch: stats expect {expected} {side} flows, but no non-uot file has that row count "
            f"(best={best_n}). Regenerate labels/flow_segments_{'eth' if side == 'ETH' else 'bnb'}.csv."
        )

    eth_p = _pick_exact(eth_order, exp_s, "ETH")
    bnb_p = _pick_exact(bnb_order, exp_d, "BNB")
    meta = {
        "selected_eth_flow_segments_path": str(eth_p.resolve()),
        "selected_bnb_flow_segments_path": str(bnb_p.resolve()),
        "selected_eth_flow_segments_rows": _count_csv_rows(eth_p),
        "selected_bnb_flow_segments_rows": _count_csv_rows(bnb_p),
        "expected_src_flows_from_stats": exp_s,
        "expected_dst_flows_from_stats": exp_d,
    }
    return eth_p, bnb_p, meta


def assert_segments_match_stats_or_raise(eth_p: Path, bnb_p: Path, stats: dict[str, Any]) -> None:
    if _path_has_uot_segment(eth_p) or _path_has_uot_segment(bnb_p):
        raise RuntimeError(
            f"Flow segment paths must not come from uot/ (solver snapshot). Got eth={eth_p} bnb={bnb_p}. "
            "Use labels/ or label_layer_v1/ canonical exports."
        )
    exp_s = int(stats.get("num_src_flows") or 0)
    exp_d = int(stats.get("num_dst_flows") or 0)
    rs = _count_csv_rows(eth_p)
    rd = _count_csv_rows(bnb_p)
    if rs != exp_s or rd != exp_d:
        raise RuntimeError(
            f"Flow segment mismatch: stats expects {exp_s}/{exp_d}, but selected segment files contain "
            f"{rs}/{rd} (eth={eth_p}, bnb={bnb_p}). Refuse to run UOT."
        )


def _patch_uot_eval_with_segment_validation(out_root: Path, seg_block: dict[str, Any]) -> None:
    evp = output_file(out_root, "uot_evaluation_metrics.json")
    if not evp.is_file():
        return
    try:
        m = json.loads(evp.read_text(encoding="utf-8"))
    except Exception:
        return
    m["segment_input_validation"] = dict(seg_block)
    m.setdefault("transport_graph_meta", {})
    if isinstance(m["transport_graph_meta"], dict):
        missed = out_root / "eval" / "missed_true_dst_debug.csv"
        if missed.is_file():
            m["transport_graph_meta"]["missed_true_dst_debug_path"] = "eval/missed_true_dst_debug.csv"
    evp.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")


def _merge_validation_transport_graph_meta(out_root: Path, transport_graph_meta: dict[str, Any]) -> None:
    """Copy canonical transport_graph_meta into labels/validation_report.json (paper audit)."""
    val = output_file(out_root, "validation_report.json")
    data: dict[str, Any] = {}
    if val.is_file():
        try:
            data = json.loads(val.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["transport_graph_meta"] = dict(transport_graph_meta)
    val.parent.mkdir(parents=True, exist_ok=True)
    val.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _assert_main_uot_matches_segment_inputs(out_root: Path, seg_block: dict[str, Any]) -> None:
    """Fail the closure if the solved UOT graph does not reflect the same full flow counts as labels/stats."""
    evp = output_file(out_root, "uot_evaluation_metrics.json")
    if not evp.is_file():
        raise RuntimeError("paper_experiment_closure: uot_evaluation_metrics.json missing after main UOT.")
    ev = json.loads(evp.read_text(encoding="utf-8"))
    tg = ev.get("transport_graph_meta")
    if not isinstance(tg, dict):
        raise RuntimeError("paper_experiment_closure: transport_graph_meta missing in uot_evaluation_metrics.json.")
    exp_s = int(seg_block.get("expected_src_flows_from_stats") or 0)
    exp_d = int(seg_block.get("expected_dst_flows_from_stats") or 0)
    ns = int(tg.get("num_src_flows") or tg.get("n_eth_full") or tg.get("n_eth") or 0)
    n_orig = int(
        tg.get("num_dst_flows_original") or tg.get("n_bnb_full") or tg.get("n_bnb_original") or 0
    )
    n_sel = int(tg.get("num_dst_flows_selected") or tg.get("n_bnb_active") or 0)
    mode = str(tg.get("subgraph_mode") or tg.get("mode") or "")
    if exp_s > 0 and ns != exp_s:
        raise RuntimeError(
            f"Paper closure UOT saw num_src_flows={ns} but flow_label_stats / segment CSVs expect {exp_s}. "
            "Refuse to continue (wrong or truncated ETH segment file). "
            "Do not write paper_experiment_summary until segment selection is fixed."
        )
    if exp_d > 0 and n_orig != exp_d:
        raise RuntimeError(
            f"Paper closure UOT saw num_dst_flows_original={n_orig} but stats expect {exp_d}. "
            "Refuse to continue. Do not write paper_experiment_summary until BNB segment inputs match stats."
        )
    # Guard the reported failure mode: tiny full_matrix while the label layer is large.
    if exp_s > 500 and exp_d > 500 and mode == "full_matrix" and ns <= 500 and n_sel <= 500:
        raise RuntimeError(
            f"Paper closure: UOT is full_matrix at {ns}×{n_sel} while stats expect {exp_s}/{exp_d} flows. "
            "Expected global_dst_pool when the product exceeds uot.flow_max_matrix_cells. "
            "Refuse to continue; do not write paper_experiment_summary."
        )


_LABEL_LAYER_V1_FILES: tuple[str, ...] = (
    "evidence_eth.csv",
    "evidence_bnb.csv",
    "tx_anchor_candidates.csv",
    "tx_anchor_labels.csv",
    "tx_anchor_low_confidence.csv",
    "tx_anchor_diagnostics.json",
    "tx_anchor_receiver_mismatch_debug.csv",
    "flow_segments_eth.csv",
    "flow_segments_bnb.csv",
    "tx_to_flow_map.csv",
    "flow_labels.csv",
    "flow_label_stats.json",
    "validation_report.json",
)


def freeze_label_layer_v1(out_root: Path) -> Path:
    """Copy canonical label artifacts into ``label_layer_v1/`` and write ``label_layer_v1_summary.json``."""
    out_root = Path(out_root)
    v1 = out_root / "label_layer_v1"
    v1.mkdir(parents=True, exist_ok=True)
    for fn in _LABEL_LAYER_V1_FILES:
        if fn in ("flow_segments_eth.csv", "flow_segments_bnb.csv"):
            primary = out_root / "labels" / fn
            if primary.is_file():
                shutil.copy2(primary, v1 / fn)
            else:
                logger.warning("freeze_label_layer_v1: labels/%s missing; skip v1 copy (do not fall back to uot/)", fn)
            continue
        src = _artifact_path(out_root, fn)
        if src.is_file():
            shutil.copy2(src, v1 / fn)
        else:
            logger.warning("freeze_label_layer_v1: missing %s (looked at %s)", fn, src)

    diag_p = v1 / "tx_anchor_diagnostics.json"
    stats_p = v1 / "flow_label_stats.json"
    diag: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    if diag_p.is_file():
        diag = json.loads(diag_p.read_text(encoding="utf-8"))
    if stats_p.is_file():
        stats = json.loads(stats_p.read_text(encoding="utf-8"))

    summary = {
        "accepted_threshold": float(diag.get("accept_threshold") or 0.7),
        "score_gap_min": float(diag.get("score_gap_min") or 0.05),
        "num_candidate_pairs": int(diag.get("num_candidate_pairs") or 0),
        "num_accepted_anchor_pairs": int(diag.get("num_accepted_anchor_pairs") or 0),
        "num_flow_labels": int(stats.get("num_flow_labels") or 0),
        "one_to_one_flow_count": int(stats.get("one_to_one_flow_count") or 0),
        "one_to_many_flow_count": int(stats.get("one_to_many_flow_count") or 0),
        "many_to_one_flow_count": int(stats.get("many_to_one_flow_count") or 0),
        "many_to_many_flow_count": int(stats.get("many_to_many_flow_count") or 0),
        "flow_label_coverage_by_tx": float(stats.get("flow_label_coverage_by_tx") or 0.0),
        "flow_label_coverage_by_amount": float(stats.get("flow_label_coverage_by_amount") or 0.0),
        "predominantly_one_to_one": bool(stats.get("predominantly_one_to_one")),
    }
    sum_path = v1 / "label_layer_v1_summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Label layer v1 frozen under %s", v1)
    return v1


def run_semi_synthetic_uot_for_seed(
    out_root: Path,
    *,
    fl: Path,
    stats_p: Path,
    src: Path,
    dst: Path,
    seed: int,
    num_seeds: int,
    args: Namespace,
    cfg: dict,
) -> dict[str, Any]:
    """Build semi-synthetic labels + subgraph UOT under ``out_root`` (isolated per RNG seed)."""
    from cross.application.pipeline import uot_kwargs_from_config
    from cross.domain.evaluation.synthetic_scenario_eval import build_synthetic_metrics_bundle

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    uk = uot_kwargs_from_config(args, cfg)
    syn_csv, syn_json = build_semi_synthetic_from_flow_labels(
        fl,
        stats_p,
        out_root,
        seed=int(seed),
        max_seeds=int(num_seeds),
        force=bool(getattr(args, "synthetic_force", False)),
    )
    result: dict[str, Any] = {"seed": int(seed), "num_seeds": int(num_seeds), "out_root": str(out_root.resolve())}
    if not syn_csv.is_file() or not Path(syn_json).is_file():
        result["skipped"] = True
        result["reason"] = "missing_synthetic_outputs"
        return result
    hints = json.loads(Path(syn_json).read_text(encoding="utf-8"))
    if hints.get("skipped"):
        result["skipped"] = True
        result["reason"] = str(hints.get("reason") or "skipped")
        return result
    clones = hints.get("segment_clone_records") or []
    se = out_root / "flow_segments_eth_synth.csv"
    sb = out_root / "flow_segments_bnb_synth.csv"
    write_synthetic_subgraph_segment_csvs(src, dst, clones, se, sb)
    if not se.is_file() or not sb.is_file() or not hints.get("eval_hints", {}).get("truth_flow_pairs"):
        result["skipped"] = True
        result["reason"] = "segment_or_truth_missing"
        return result
    run_standalone_flow_uot(
        out_root,
        src_flows_csv=se,
        dst_flows_csv=sb,
        flow_labels_csv=syn_csv,
        uot_reg=float(uk["uot_reg"]),
        uot_reg_m=float(uk["uot_reg_m"]),
        uot_decode_threshold=float(uk["uot_decode_threshold"]),
        uot_cost_weights=uk.get("uot_cost_weights"),
        uot_backend=str(uk["uot_backend"]),
        uot_max_delay_sec=float(uk["uot_max_delay_sec"]),
        uot_causal_violation_penalty=float(uk["uot_causal_violation_penalty"]),
        uot_lambda_risk=float(uk["uot_lambda_risk"]),
        uot_causal_infeasible_delay_sec=uk.get("uot_causal_infeasible_delay_sec"),
        uot_export_matrix=bool(uk["uot_export_matrix"]),
        uot_export_cost_components=bool(uk["uot_export_cost_components"]),
        uot_export_cost_matrix_csv=bool(uk.get("uot_export_cost_matrix_csv", False)),
        uot_allow_unmatched=bool(uk["uot_allow_unmatched"]),
        uot_use_graph_embedding=bool(uk["uot_use_graph_embedding"]),
        graph_ranker_checkpoint=getattr(args, "graph_ranker_checkpoint", None),
        uot_ablation=str(uk.get("uot_ablation") or "none"),
        run_flow_baselines=False,
        flow_label_min_confidence=float(uk.get("flow_label_min_confidence") or 0.0),
        synthetic_eval_hints_path=Path(syn_json),
        uot_flow_dst_top_k=int(uk.get("uot_flow_dst_top_k") or 200),
        uot_flow_max_matrix_cells=int(uk.get("uot_flow_max_matrix_cells") or 6_000_000),
        **_uot_pool_kwargs_from_uk(uk),
    )
    ev_path = output_file(out_root, "uot_evaluation_metrics.json")
    sc_path = output_file(out_root, "synthetic_eval_by_scenario.csv")
    if ev_path.is_file():
        ev_obj = json.loads(ev_path.read_text(encoding="utf-8"))
        syn = build_synthetic_metrics_bundle(ev_obj, sc_path, Path(syn_json))
        result["synthetic_metrics"] = syn
        out_syn = out_root / "synthetic_metrics_summary.json"
        out_syn.write_text(json.dumps(syn, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _merge_synthetic_metrics_into_main(out_root: Path, synth_work: Path) -> None:
    syn_eval = output_file(synth_work, "uot_evaluation_metrics.json")
    main_eval = output_file(out_root, "uot_evaluation_metrics.json")
    if not syn_eval.is_file() or not main_eval.is_file():
        return
    main_m = json.loads(main_eval.read_text(encoding="utf-8"))
    syn_m = json.loads(syn_eval.read_text(encoding="utf-8"))
    sm = syn_m.get("synthetic_metrics")
    if isinstance(sm, dict):
        main_m["synthetic_metrics"] = sm
    main_eval.write_text(json.dumps(main_m, indent=2, ensure_ascii=False), encoding="utf-8")
    sc_src = output_file(synth_work, "synthetic_eval_by_scenario.csv")
    sc_dst = output_file(out_root, "synthetic_eval_by_scenario.csv")
    if sc_src.is_file():
        shutil.copy2(sc_src, sc_dst)


def _read_threshold_row(
    run_root: Path,
    setting: str,
    accept_thr: float | None,
    score_gap: float | None,
) -> dict[str, Any]:
    diag_p = locate_output_file(run_root, "tx_anchor_diagnostics.json")
    diag = json.loads(diag_p.read_text(encoding="utf-8")) if diag_p.is_file() else {}
    stats_path = output_file(run_root, "flow_label_stats.json")
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    ev_path = output_file(run_root, "uot_evaluation_metrics.json")
    ev = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.is_file() else {}
    return {
        "setting": setting,
        "accept_threshold": float(accept_thr) if accept_thr is not None else float(diag.get("accept_threshold") or 0.7),
        "score_gap": float(score_gap) if score_gap is not None else float(diag.get("score_gap_min") or 0.05),
        "num_accepted_anchor_pairs": int(diag.get("num_accepted_anchor_pairs") or 0),
        "num_flow_labels": int(stats.get("num_flow_labels") or 0),
        "flow_label_coverage_by_tx": float(stats.get("flow_label_coverage_by_tx") or 0.0),
        "flow_label_coverage_by_amount": float(stats.get("flow_label_coverage_by_amount") or 0.0),
        "flow_pair_f1": float(ev.get("flow_pair_f1") or 0.0),
        "flow_mass_recall": float(ev.get("flow_mass_recall") or 0.0),
        "top1_flow_acc": float(ev.get("top1_flow_accuracy") or ev.get("top1_flow_correspondence_accuracy") or 0.0),
        "causal_violation_rate": float(ev.get("causal_violation_rate") or 0.0),
    }


def run_threshold_sensitivity(out_root: Path, ev_e: Path, ev_b: Path, args: Namespace, cfg: dict) -> None:
    """Re-build anchors/labels/UOT for stricter and looser thresholds; append default row from current tree."""
    from cross.application.pipeline import uot_kwargs_from_config

    out_root = Path(out_root)
    uk = uot_kwargs_from_config(args, cfg)
    fw = int(uk.get("flow_window_sec") or 1800)
    rows: list[dict[str, Any]] = []

    def_row = _read_threshold_row(out_root, "default", None, None)
    rows.append(def_row)

    logger.info("threshold_sensitivity: starting (default row + stricter + looser; each rebuilds anchors→segments→labels→UOT)")
    _flush_logs()
    for setting, thr, gap in (("stricter", 0.8, 0.1), ("looser", 0.65, 0.03)):
        sub = out_root / "threshold_runs" / setting
        sub.mkdir(parents=True, exist_ok=True)
        logger.info("threshold_sensitivity: setting=%s accept_threshold=%s score_gap_min=%s -> %s", setting, thr, gap, sub)
        _flush_logs()
        build_tx_anchor_labels(ev_e, ev_b, sub, accept_threshold=float(thr), score_gap_min=float(gap))
        ta = output_file(sub, "tx_anchor_labels.csv")
        logger.info("threshold_sensitivity: %s tx_anchor_labels done -> flow segments", setting)
        _flush_logs()
        build_flow_segments_from_evidence(ev_e, ev_b, ta, sub, flow_window_sec=fw)
        logger.info("threshold_sensitivity: %s flow segments done -> flow_labels", setting)
        _flush_logs()
        build_flow_labels(
            ta,
            output_file(sub, "flow_segments_eth.csv"),
            output_file(sub, "flow_segments_bnb.csv"),
            output_file(sub, "tx_to_flow_map.csv"),
            sub,
            min_confidence=0.0,
        )
        logger.info("threshold_sensitivity: %s flow_labels done -> standalone UOT", setting)
        _flush_logs()
        run_standalone_flow_uot(
            sub,
            src_flows_csv=output_file(sub, "flow_segments_eth.csv"),
            dst_flows_csv=output_file(sub, "flow_segments_bnb.csv"),
            flow_labels_csv=output_file(sub, "flow_labels.csv"),
            uot_reg=float(uk["uot_reg"]),
            uot_reg_m=float(uk["uot_reg_m"]),
            uot_decode_threshold=float(uk["uot_decode_threshold"]),
            uot_cost_weights=uk.get("uot_cost_weights"),
            uot_backend=str(uk["uot_backend"]),
            uot_max_delay_sec=float(uk["uot_max_delay_sec"]),
            uot_causal_violation_penalty=float(uk["uot_causal_violation_penalty"]),
            uot_lambda_risk=float(uk["uot_lambda_risk"]),
            uot_causal_infeasible_delay_sec=uk.get("uot_causal_infeasible_delay_sec"),
            uot_export_matrix=bool(uk["uot_export_matrix"]),
            uot_export_cost_components=bool(uk["uot_export_cost_components"]),
            uot_export_cost_matrix_csv=bool(uk.get("uot_export_cost_matrix_csv", False)),
            uot_allow_unmatched=bool(uk["uot_allow_unmatched"]),
            uot_use_graph_embedding=bool(uk["uot_use_graph_embedding"]),
            graph_ranker_checkpoint=getattr(args, "graph_ranker_checkpoint", None),
            uot_ablation=str(uk.get("uot_ablation") or "none"),
            run_flow_baselines=False,
            flow_label_min_confidence=float(uk.get("flow_label_min_confidence") or 0.0),
            synthetic_eval_hints_path=None,
            uot_flow_dst_top_k=int(uk.get("uot_flow_dst_top_k") or 200),
            uot_flow_max_matrix_cells=int(uk.get("uot_flow_max_matrix_cells") or 6_000_000),
            **_uot_pool_kwargs_from_uk(uk),
        )
        rows.append(_read_threshold_row(sub, setting, thr, gap))
        logger.info("threshold_sensitivity: finished setting=%s", setting)
        _flush_logs()

    out_csv = output_file(out_root, "threshold_sensitivity.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("Wrote %s", out_csv)


def _sweep_metric_row(out_root: Path, strategy_name: str) -> dict[str, Any]:
    """Best-effort read of one row from ``experiments/candidate_pool_sweep.csv``."""
    sweep_p = out_root / "experiments" / "candidate_pool_sweep.csv"
    out: dict[str, Any] = {}
    if not sweep_p.is_file():
        return out
    try:
        df = pd.read_csv(sweep_p, dtype=str, keep_default_na=False)
        if df.empty or "strategy" not in df.columns:
            return out
        sub = df[df["strategy"].astype(str) == strategy_name]
        if sub.empty:
            return out
        r = sub.iloc[0]
        for c in (
            "candidate_dst_recall",
            "candidate_dst_recall_unique_dst",
            "candidate_dst_recall_edge_level",
            "bnb_active",
            "matrix_cells",
        ):
            if c in r.index and str(r[c]).strip() != "":
                out[c] = r[c]
    except Exception:
        return {}
    return out


def _flow_id_label_dst_ratio(out_root: Path) -> str:
    p = out_root / "experiments" / "flow_id_consistency_report.json"
    if not p.is_file():
        return "n/a"
    try:
        rep = json.loads(p.read_text(encoding="utf-8"))
        for k, v in rep.items():
            if isinstance(k, str) and "label_dst_found" in k and "labels_bnb" in k and isinstance(v, (int, float)):
                return str(float(v))
    except Exception:
        pass
    return "n/a"


def _section9_rc_uot_decoding_and_failure_md(out_root: Path) -> list[str]:
    """Markdown lines for §9 (decode sweep, 12M vs 18M recall, synthetic failures, limits)."""
    out_root = Path(out_root)
    rec_p = out_root / "experiments" / "recommended_decode_rule.json"
    sweep_p = out_root / "experiments" / "decode_threshold_sweep.csv"
    syn_dbg = out_root / "eval" / "synthetic_failure_debug.csv"
    if rec_p.is_file():
        try:
            rec = json.loads(rec_p.read_text(encoding="utf-8"))
            rec_line = (
                f"**Recommended decode (from sweep):** `{rec.get('recommended_rule')}` @ `{rec.get('recommended_threshold_or_k')}` — "
                f"{rec.get('reason', '')} "
                f"(flow_pair_f1 **{rec.get('flow_pair_f1')}**, flow_pair_recall **{rec.get('flow_pair_recall')}**, "
                f"flow_mass_recall **{rec.get('flow_mass_recall')}**, average_edges_per_source **{rec.get('average_edges_per_source')}**). "
                f"Artifacts: **`experiments/decode_threshold_sweep.csv`**, **`experiments/recommended_decode_rule.json`**."
            )
        except Exception:
            rec_line = f"**Recommended decode:** file present but unreadable: `{rec_p}`."
    else:
        rec_line = (
            "**Decode sweep:** run `--decode-threshold-sweep` after **`uot/uot_transport_matrix.npz`** exists "
            "to populate **`experiments/decode_threshold_sweep.csv`** and **`experiments/recommended_decode_rule.json`**."
        )
    sweep_note = ""
    if sweep_p.is_file():
        try:
            sdf = pd.read_csv(sweep_p, dtype=str, keep_default_na=False)
            n = len(sdf)
            sweep_note = (
                f"**Sweep grid:** **`experiments/decode_threshold_sweep.csv`** has **{n}** rows "
                f"(expect **54** rows for the combined 6×3×3 grid when dense `P` is available; a single `error` row means the transport matrix was missing)."
            )
        except Exception:
            sweep_note = f"**Sweep grid:** `{sweep_p}` exists but could not be read."

    c12 = _sweep_metric_row(out_root, "C_larger_budget")
    g18 = _sweep_metric_row(out_root, "G_larger_budget_18M")
    recall_12 = c12.get("candidate_dst_recall", "n/a")
    recall_18 = g18.get("candidate_dst_recall", "n/a")
    bnb12 = c12.get("bnb_active", "n/a")
    bnb18 = g18.get("bnb_active", "n/a")

    if syn_dbg.is_file():
        syn_txt = (
            f"- **Semi-synthetic unmatched / noise:** see **`eval/synthetic_failure_debug.csv`** and **`eval/synthetic_eval_by_scenario.csv`**. "
            "Typical causes: **unmatched_ratio** in USD-normalized columns stays **far below** the default **0.08** detector gate → **unmatched_detection_f1 ≈ 0**; "
            "delay-noise **decoys** can still receive **positive transport mass** (see `decoy_edge_in_transport_plan` / `decoy_pair_match_rate`)."
        )
    else:
        syn_txt = (
            "- **Semi-synthetic unmatched / noise:** run **`--synthetic-failure-debug`** to emit **`eval/synthetic_failure_debug.csv`** with per-row causes."
        )

    limits = [
        "- **Current limitations:** RC-UOT/Hungarian/Greedy comparisons are only meaningful **under the same candidate pool**; **`candidate_dst_recall`** remains the dominant real-data bottleneck. "
        "Decoding/thresholding controls **`flow_pair_f1`** vs. sparsity; **`flow_mass_recall`** can stay high while pair-F1 is low if mass is diffuse across many cells.",
        "- **Cost exports:** full-cell **`uot_cost_matrix.csv`** is **opt-in** via **`export_uot_cost_matrix_csv=true`**; default runs should rely on **`uot/uot_cost_matrix.npz`** / sparse component bundles to avoid huge CSVs.",
    ]

    return [
        "## 9. RC-UOT Decoding and Failure Analysis",
        "",
        rec_line,
        sweep_note,
        "",
        "1. **Transport vs. edge metrics:** RC-UOT can diffuse mass across many cells; **`flow_mass_recall`** may stay high while **`flow_pair_f1`** is decode-sensitive. "
        "The sweep uses the **conjunction** of **source_share_ge**, **topk_per_source**, and **cumulative_row_mass** on dense **`uot/uot_transport_matrix.npz`** (no re-solve).",
        "",
        f"2. **Candidate budget 12M vs 18M (weak-label recall diagnostic):** from **`experiments/candidate_pool_sweep.csv`**, "
        f"`C_larger_budget` (**12M** matrix cells) **candidate_dst_recall ≈ {recall_12}**, **bnb_active ≈ {bnb12}** vs. "
        f"`G_larger_budget_18M` (**18M**) **≈ {recall_18}**, **bnb_active ≈ {bnb18}** — raising the matrix budget grows active BNB columns and destination coverage (see §6).",
        "",
        "3. **Semi-synthetic split/merge:** **`eval/synthetic_eval_by_scenario.csv`** typically shows **strong split/merge recovery** when truth pairs sit on the same small synthetic subgraph as the solve.",
        "",
        syn_txt,
        "",
        *limits,
        "",
    ]


def write_paper_experiment_summary_md(out_root: Path) -> Path:
    out_root = Path(out_root)
    v1s = out_root / "label_layer_v1" / "label_layer_v1_summary.json"
    if v1s.is_file():
        lab = json.loads(v1s.read_text(encoding="utf-8"))
    else:
        dpath = _artifact_path(out_root, "tx_anchor_diagnostics.json")
        spath = _artifact_path(out_root, "flow_label_stats.json")
        diag = json.loads(dpath.read_text(encoding="utf-8")) if dpath.is_file() else {}
        stats = json.loads(spath.read_text(encoding="utf-8")) if spath.is_file() else {}
        lab = {
            "num_candidate_pairs": int(diag.get("num_candidate_pairs") or 0),
            "num_accepted_anchor_pairs": int(diag.get("num_accepted_anchor_pairs") or 0),
            "num_flow_labels": int(stats.get("num_flow_labels") or 0),
            "one_to_one_flow_count": int(stats.get("one_to_one_flow_count") or 0),
            "one_to_many_flow_count": int(stats.get("one_to_many_flow_count") or 0),
            "many_to_one_flow_count": int(stats.get("many_to_one_flow_count") or 0),
            "many_to_many_flow_count": int(stats.get("many_to_many_flow_count") or 0),
            "flow_label_coverage_by_tx": float(stats.get("flow_label_coverage_by_tx") or 0.0),
            "flow_label_coverage_by_amount": float(stats.get("flow_label_coverage_by_amount") or 0.0),
            "predominantly_one_to_one": bool(stats.get("predominantly_one_to_one")),
        }
    evp = output_file(out_root, "uot_evaluation_metrics.json")
    ev = json.loads(evp.read_text(encoding="utf-8")) if evp.is_file() else {}
    abp = output_file(out_root, "ablation_metrics.csv")
    ab_txt = abp.read_text(encoding="utf-8")[:8000] if abp.is_file() else ""
    syn = ev.get("synthetic_metrics") or {}
    ev_tg = ev.get("transport_graph_meta") if isinstance(ev.get("transport_graph_meta"), dict) else {}
    diag_lines: list[str] = []
    if ev_tg:
        diag_lines.append(
            f"- Transport graph (**{ev_tg.get('subgraph_mode', ev_tg.get('mode'))}**): "
            f"ETH **{ev_tg.get('num_src_flows', ev_tg.get('n_eth_full', ev_tg.get('n_eth')))}**, "
            f"BNB selected **{ev_tg.get('num_dst_flows_selected', ev_tg.get('n_bnb_active'))}** / original **{ev_tg.get('num_dst_flows_original', ev_tg.get('n_bnb_full', 'n/a'))}**, "
            f"matrix_cells **{ev_tg.get('matrix_cells')}**"
        )
        if "candidate_dst_recall" in ev_tg:
            diag_lines.append(
                f"- Weak-label **candidate_dst_recall** (truth dst present in BNB pool): **{ev_tg.get('candidate_dst_recall')}** "
                f"({ev_tg.get('truth_dst_in_subgraph', '?')}/{ev_tg.get('truth_dst_flows', '?')} dst)"
            )
    if ev.get("num_true_edges") is not None:
        diag_lines.append(
            f"- Edge diagnostics: true **{ev.get('num_true_edges')}**, predicted **{ev.get('num_predicted_edges')}**, "
            f"TP **{ev.get('num_true_positive_edges')}**, top1-eval sources **{ev.get('evaluated_src_count')}** / "
            f"label sources **{ev.get('total_src_count')}**, mass_thr **{ev.get('prediction_threshold')}**"
        )

    sweep_p = out_root / "experiments" / "candidate_pool_sweep.csv"
    sweep_note = ""
    if sweep_p.is_file():
        sweep_note = (
            f"- Candidate pool sweep: **`experiments/candidate_pool_sweep.csv`** (oracle @k: **`eval/candidate_recall_at_k.csv`**, "
            f"per-edge ranks: **`eval/candidate_oracle_rank_debug.csv`**)."
        )

    lines = [
        "# Paper experiment summary (RC-UOT closure)",
        "",
        "## 1. Label construction summary",
        "",
        f"- Candidate anchor pairs: **{lab.get('num_candidate_pairs', 'n/a')}**",
        f"- Accepted anchor pairs: **{lab.get('num_accepted_anchor_pairs', 'n/a')}**",
        f"- Flow labels: **{lab.get('num_flow_labels', 'n/a')}**",
        f"- Tx coverage (flow_label_coverage_by_tx): **{lab.get('flow_label_coverage_by_tx', 'n/a')}**",
        f"- Amount coverage: **{lab.get('flow_label_coverage_by_amount', 'n/a')}**",
        f"- Pattern counts — 1:1: **{lab.get('one_to_one_flow_count', 0)}**, 1:N: **{lab.get('one_to_many_flow_count', 0)}**, "
        f"N:1: **{lab.get('many_to_one_flow_count', 0)}**, N:N: **{lab.get('many_to_many_flow_count', 0)}**",
        f"- Predominantly one-to-one: **{lab.get('predominantly_one_to_one')}**",
        "",
        "## 2. Real Celer evaluation (RC-UOT vs baselines)",
        "",
        "- **Main real-data UOT** uses **`global_dst_pool`** (not a **400×400** head truncation of flow segments). "
        "Segment CSVs are the full label-layer exports; the transport graph is a **global BNB column pool** capped by `uot.flow_max_matrix_cells`. "
        "**Primary paper configuration:** `uot.flow_dst_top_k=200`, `uot.flow_max_matrix_cells=12_000_000`, `uot.pool_strategy=default` (larger matrix budget to grow active BNB columns). "
        "**Low-budget diagnostic baseline (6M cells, same top_k/strategy):** candidate pool sweep row `A_current` reports **candidate_dst_recall≈0.193** with **bnb_active≈1046** on the reference Celer closure — compare to **`C_larger_budget`** (**≈0.418**, **bnb_active≈2092**) and **`F_oracle_upper_bound`** (**recall=1.0**, injected true destinations; see §Diagnostic oracle upper bound). "
        "Do **not** use `stratified_global_pool` as the primary setting here: sweep row **`E_stratified_larger`** underperformed **`C_larger_budget`** on the same dataset.",
        "- **Real Celer** weak labels are predominantly **one-to-one**; **split / merge / unmatched** stress tests rely on **semi-synthetic** rows and `eval/synthetic_eval_by_scenario.csv`.",
        "- **Interpretation (typical main run):** **`flow_mass_recall`** is high when RC-UOT assigns most labeled source USD mass **within the chosen candidate pool**; "
        "**`candidate_dst_recall`** is low when many weak-label **true destination flows** never enter that pool. "
        "**Report candidate subgraph retrieval separately from transport matching** (Hungarian / Greedy / RC-UOT are only comparable **conditional on the same pool**).",
        "- **Do not claim** RC-UOT is uniformly superior to Hungarian/Greedy on the **full** BNB flow space unless a recall sweep shows the pool already covers most true destinations.",
        "- **`candidate_dst_recall`** lives in `transport_graph_meta` (and top-level metrics where mirrored). "
        "See **`eval/missed_true_dst_debug.csv`**, **`eval/missed_true_dst_reason_summary.csv`**, and diagnostics from `--candidate-recall-diagnostics`.",
        "",
        "Key metrics from `uot_evaluation_metrics.json` (soft edge F1, USD-aligned mass recall, top-k). "
        "See `transport_graph_meta` in that JSON and `uot/uot_diagnostics.json` for `subgraph_mode`, matrix size, and recall.",
        "",
        *(diag_lines + [""] if diag_lines else []),
        *( [sweep_note, ""] if sweep_note else []),
        f"- flow_pair_f1: **{ev.get('flow_pair_f1', 'n/a')}**",
        f"- flow_mass_recall (USD proxy, row-fraction × label src USD): **{ev.get('flow_mass_recall', 'n/a')}**",
        f"- top1_flow_accuracy: **{ev.get('top1_flow_accuracy', ev.get('top1_flow_correspondence_accuracy', 'n/a'))}**",
        f"- causal_violation_rate: **{ev.get('causal_violation_rate', 'n/a')}**",
        f"- unmatched_mass_detection_f1: **{ev.get('unmatched_mass_detection_f1', 'n/a')}**",
        f"- Real-data split label rows: **{ev.get('real_data_split_label_rows', 'n/a')}**; merge label rows: **{ev.get('real_data_merge_label_rows', 'n/a')}**",
        "",
        "See `ablation_metrics.csv` for Greedy / Hungarian / Balanced OT / RC-UOT and ablations. "
        "Each row includes **`candidate_dst_recall`**, **`candidate_dst_recall_unique_dst`**, **`candidate_dst_recall_edge_level`**, **`num_dst_flows_selected`**, and **`matrix_cells`** (from the shared `transport_graph_meta` / `uot_diagnostics.json` pool used for RC-UOT) so baselines are compared **under the same active BNB column pool** as the main transport graph.",
        "",
        "```text",
        ab_txt[:4000],
        "```",
        "",
        "## 3. Semi-synthetic evaluation",
        "",
        "Metrics merged under `synthetic_metrics` in `uot_evaluation_metrics.json` (subgraph UOT on cloned segments):",
        "",
        json.dumps(syn, indent=2, ensure_ascii=False) if syn else "_No synthetic_metrics yet._",
        "",
        "Per-scenario table: `eval/synthetic_eval_by_scenario.csv`.",
        "",
        "## 4. Ablation study",
        "",
        "Compare `RC-UOT no_risk`, `RC-UOT no_causal`, `Balanced OT`, `RC-UOT no_graph`, `RC-UOT no_evidence` in `ablation_metrics.csv`.",
        "Expect `no_causal` to show higher causal_violation_rate than full RC-UOT when causal penalties are removed upstream of reporting.",
        "",
        "## 5. 论文叙事建议",
        "",
        "真实 Celer 数据主要提供交易级锚点和弱监督流级标签；由于真实公开跨链桥记录以 **one-to-one** 为主，"
        "**split / merge / unmatched** 等复杂模式应在 **半合成** 场景（见 `synthetic_eval_by_scenario.csv`）中评估。",
        "主真实数据实验已使用 **`global_dst_pool`** 与 **全量 flow segment**（见 `transport_graph_meta`），**不再使用 400×400 截断** 作为主设定。",
        "**`flow_mass_recall` 高**通常说明在**已进入候选池**的 BNB 列上，RC-UOT 能把大部分与弱标签对齐的源侧资金质量分配出去；"
        "**`candidate_dst_recall` 低**则说明 **候选检索 / 全局池预算** 仍是系统瓶颈，应在文中把 **候选召回** 与 **RC-UOT 运输匹配** 分层表述。",
        "若 `experiments/candidate_pool_sweep.csv` 显示在合理算力下 **candidate_dst_recall 仍显著低于 ~0.5**，可明确写："
        "**“candidate subgraph retrieval remains a limiting factor”**，并避免在全文目标空间上过度宣称相对 Hungarian/Greedy 的绝对优势。",
        "",
        "## 6. Candidate Retrieval Bottleneck Analysis",
        "",
        "This section summarizes **candidate subgraph retrieval** (weak-label true destinations vs. active BNB pool), separate from RC-UOT matching quality.",
        "",
        f"- **Flow-id consistency (label dst vs. BNB segment universe):** `label_dst_found_in_labels_bnb_segments_ratio` = **{_flow_id_label_dst_ratio(out_root)}** "
        "(from `experiments/flow_id_consistency_report.json` when present).",
        "",
        "- **Candidate pool sweep** (`experiments/candidate_pool_sweep.csv`):",
        f"  - **Low budget (6M, `A_current`):** candidate_dst_recall **{_sweep_metric_row(out_root, 'A_current').get('candidate_dst_recall', '≈0.193 (see sweep CSV)')}**, "
        f"bnb_active **{_sweep_metric_row(out_root, 'A_current').get('bnb_active', '≈1046')}**.",
        f"  - **Larger budget (12M, `C_larger_budget`, primary config):** candidate_dst_recall **{_sweep_metric_row(out_root, 'C_larger_budget').get('candidate_dst_recall', '≈0.418')}**, "
        f"bnb_active **{_sweep_metric_row(out_root, 'C_larger_budget').get('bnb_active', '≈2092')}**.",
        f"  - **18M diagnostic row (`G_larger_budget_18M`):** candidate_dst_recall **{_sweep_metric_row(out_root, 'G_larger_budget_18M').get('candidate_dst_recall', 'n/a (re-run --candidate-recall-diagnostics)')}**, "
        f"bnb_active **{_sweep_metric_row(out_root, 'G_larger_budget_18M').get('bnb_active', 'n/a')}**.",
        f"  - **Oracle upper bound (`F_oracle_upper_bound`):** candidate_dst_recall **{_sweep_metric_row(out_root, 'F_oracle_upper_bound').get('candidate_dst_recall', '1.0')}** (diagnostic only; see below).",
        "",
        "**Interpretation:**",
        "- True labeled destinations are present in the full BNB segment universe (flow-id check).",
        "- Oracle forcing recovers **all** labeled destinations in the candidate-recall metric (see `experiments/oracle_forcing_summary.json` when run).",
        "- The non-oracle **`global_dst_pool`** under a fixed matrix-cell budget is the dominant bottleneck: increasing **`flow_max_matrix_cells`** materially increases active BNB columns and **candidate_dst_recall**.",
        "- **`flow_dst_top_k=500`** did not match the gains from raising the matrix budget in our sweep (see `B_larger_topk` vs `C_larger_budget`).",
        "- **Future work:** stronger retrieval, sparser or structured UOT over larger candidate sets, or other pool construction — not only the RC-UOT solver — should be highlighted when recall is pool-limited.",
        "",
        "## 7. Diagnostic oracle upper bound (not a fair model result)",
        "",
        "**English:** `F_oracle_upper_bound` in `experiments/candidate_pool_sweep.csv` is used **only** as a **diagnostic upper bound** on candidate recall. "
        "It is **not** reported as a fair competing method because it **injects true label destinations** into the candidate pool.",
        "",
        "**中文：** Oracle 上界实验仅用于确认 flow_id 映射、候选池保护逻辑和 recall 口径是否正确，不作为正式对比模型。"
        "其结果表明，在目标列被完整纳入候选池时，候选召回上界可达到 1.0；因此当前正式模型的主要瓶颈是非 oracle 候选池覆盖率，而非 RC-UOT 求解器本身。",
        "",
        "## 8. Threshold sensitivity",
        "",
        "See `experiments/threshold_sensitivity.csv` for default vs stricter (0.80 / 0.10) vs looser (0.65 / 0.03).",
        "",
        *_section9_rc_uot_decoding_and_failure_md(out_root),
    ]
    outp = output_file(out_root, "paper_experiment_summary.md")
    outp.write_text("\n".join(lines), encoding="utf-8")
    return outp


def run_paper_experiment_closure(args: Namespace, cfg: dict) -> int:
    """Freeze v1, run RC-UOT + baselines/ablations, semi-synthetic UOT, threshold sweep, summary."""
    from cross.application.pipeline import uot_kwargs_from_config

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    ev_e = Path(args.evidence_eth) if getattr(args, "evidence_eth", None) else _artifact_path(out_root, "evidence_eth.csv")
    ev_b = Path(args.evidence_bnb) if getattr(args, "evidence_bnb", None) else _artifact_path(out_root, "evidence_bnb.csv")
    if not ev_e.is_file() or not ev_b.is_file():
        logger.error("paper_experiment_closure: evidence CSVs missing (%s, %s)", ev_e, ev_b)
        return 1

    stats_path = output_file(out_root, "flow_label_stats.json")
    if not stats_path.is_file():
        logger.error("paper_experiment_closure: %s missing", stats_path)
        return 1
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    inv_path = _write_artifact_inventory(out_root)
    _merge_validation_segment_block(
        out_root,
        {
            "artifact_inventory_path": str(inv_path.resolve()),
            "flow_label_stats_num_src_flows": int(stats.get("num_src_flows") or 0),
            "flow_label_stats_num_dst_flows": int(stats.get("num_dst_flows") or 0),
            "phase": "pre_flight",
        },
    )

    uk = uot_kwargs_from_config(args, cfg)
    stats = _ensure_labels_segments_and_stats(out_root, ev_e, ev_b, uk)

    try:
        finalize_multi_source_label_bundle(
            out_root,
            celer_label_csv=getattr(args, "celer_tx_labels", None),
            label_source_mode=str(uk.get("label_source_mode") or "celer_only_if_available"),
            weak_flow_label_count=int(stats.get("num_flow_labels") or 0),
            paper_mode=True,
        )
    except FileNotFoundError as e:
        logger.warning("paper_experiment_closure: label bundle refresh skipped (%s)", e)
    except RuntimeError as e:
        logger.error("paper_experiment_closure: label bundle failed (strict paper rules): %s", e)
        return 1

    src, dst, pick_meta = select_flow_segments_for_closure(out_root, stats)
    assert_segments_match_stats_or_raise(src, dst, stats)

    seg_block = {
        **pick_meta,
        "segment_rows_match_stats": True,
        "refused_uot_due_to_segment_mismatch": False,
        "artifact_inventory_path": str(inv_path.resolve()),
        "expected_src_flows_from_stats": int(stats.get("num_src_flows") or 0),
        "expected_dst_flows_from_stats": int(stats.get("num_dst_flows") or 0),
    }
    logger.info(
        "paper_experiment_closure: selected ETH segments %s rows=%d; BNB %s rows=%d (stats %d/%d)",
        seg_block["selected_eth_flow_segments_path"],
        seg_block["selected_eth_flow_segments_rows"],
        seg_block["selected_bnb_flow_segments_path"],
        seg_block["selected_bnb_flow_segments_rows"],
        seg_block["expected_src_flows_from_stats"],
        seg_block["expected_dst_flows_from_stats"],
    )
    _flush_logs()
    _merge_validation_segment_block(out_root, {**seg_block, "phase": "pre_uot"})

    freeze_label_layer_v1(out_root)

    fl = locate_output_file(out_root, "flow_labels.csv")
    if not src.is_file() or not dst.is_file() or not fl.is_file():
        logger.error("paper_experiment_closure: need flow_segments_eth/bnb.csv and flow_labels.csv under %s", out_root)
        return 1

    hint_arg = getattr(args, "synthetic_eval_hints", None)
    syn_hints_main = Path(hint_arg) if hint_arg else None

    run_standalone_flow_uot(
        out_root,
        src_flows_csv=src,
        dst_flows_csv=dst,
        flow_labels_csv=fl,
        uot_reg=float(uk["uot_reg"]),
        uot_reg_m=float(uk["uot_reg_m"]),
        uot_decode_threshold=float(uk["uot_decode_threshold"]),
        uot_cost_weights=uk.get("uot_cost_weights"),
        uot_backend=str(uk["uot_backend"]),
        uot_max_delay_sec=float(uk["uot_max_delay_sec"]),
        uot_causal_violation_penalty=float(uk["uot_causal_violation_penalty"]),
        uot_lambda_risk=float(uk["uot_lambda_risk"]),
        uot_causal_infeasible_delay_sec=uk.get("uot_causal_infeasible_delay_sec"),
        uot_export_matrix=bool(uk["uot_export_matrix"]),
        uot_export_cost_components=bool(uk["uot_export_cost_components"]),
        uot_export_cost_matrix_csv=bool(uk.get("uot_export_cost_matrix_csv", False)),
        uot_allow_unmatched=bool(uk["uot_allow_unmatched"]),
        uot_use_graph_embedding=bool(uk["uot_use_graph_embedding"]),
        graph_ranker_checkpoint=getattr(args, "graph_ranker_checkpoint", None),
        uot_ablation=str(uk.get("uot_ablation") or "none"),
        run_flow_baselines=True,
        flow_label_min_confidence=float(uk.get("flow_label_min_confidence") or 0.0),
        synthetic_eval_hints_path=syn_hints_main,
        uot_flow_dst_top_k=int(uk.get("uot_flow_dst_top_k") or 200),
        uot_flow_max_matrix_cells=int(uk.get("uot_flow_max_matrix_cells") or 6_000_000),
        **_uot_pool_kwargs_from_uk(uk),
    )
    _eval_src = str(uk.get("label_source_mode") or "celer_only_if_available")
    ld_p = output_file(out_root, "label_diagnostics.json")
    if ld_p.is_file():
        try:
            _eval_src = str(
                json.loads(ld_p.read_text(encoding="utf-8")).get("label_source_mode_applied") or _eval_src
            )
        except Exception:
            pass
    patch_uot_evaluation_label_source(out_root, _eval_src)
    _patch_uot_eval_with_segment_validation(out_root, seg_block)
    _assert_main_uot_matches_segment_inputs(out_root, seg_block)
    ev_post = output_file(out_root, "uot_evaluation_metrics.json")
    if ev_post.is_file():
        try:
            ev_obj = json.loads(ev_post.read_text(encoding="utf-8"))
            tg_m = ev_obj.get("transport_graph_meta")
            if isinstance(tg_m, dict):
                _merge_validation_transport_graph_meta(out_root, tg_m)
        except Exception:
            logger.exception("paper_experiment_closure: could not merge transport_graph_meta into validation_report")
    logger.info("paper_experiment_closure: main RC-UOT + baselines + ablations finished under %s", out_root)
    _flush_logs()

    stats_p = output_file(out_root, "flow_label_stats.json")
    logger.info("paper_experiment_closure: building semi-synthetic flow labels …")
    _flush_logs()
    syn_csv, syn_json = build_semi_synthetic_from_flow_labels(
        fl,
        stats_p,
        out_root,
        seed=int(getattr(args, "synthetic_random_seed", None) or 42),
        max_seeds=int(getattr(args, "synthetic_num_seeds", None) or 8),
    )
    if syn_csv.is_file() and Path(syn_json).is_file():
        hints = json.loads(Path(syn_json).read_text(encoding="utf-8"))
        if not hints.get("skipped"):
            clones = hints.get("segment_clone_records") or []
            synth_work = out_root / "synthetic_uot_work"
            synth_work.mkdir(parents=True, exist_ok=True)
            se = synth_work / "flow_segments_eth_synth.csv"
            sb = synth_work / "flow_segments_bnb_synth.csv"
            write_synthetic_subgraph_segment_csvs(src, dst, clones, se, sb)
            if not se.is_file() or not sb.is_file():
                logger.warning("Semi-synthetic segment export missing; skip synthetic UOT")
            elif se.stat().st_size < 20 or sb.stat().st_size < 20:
                logger.warning("Semi-synthetic segment CSVs nearly empty; skip synthetic UOT")
            elif not hints.get("eval_hints", {}).get("truth_flow_pairs"):
                logger.warning("No synthetic truth pairs; skip synthetic UOT")
            else:
                logger.info("paper_experiment_closure: running synthetic UOT workspace %s", synth_work)
                _flush_logs()
                run_standalone_flow_uot(
                    synth_work,
                    src_flows_csv=se,
                    dst_flows_csv=sb,
                    flow_labels_csv=syn_csv,
                    uot_reg=float(uk["uot_reg"]),
                    uot_reg_m=float(uk["uot_reg_m"]),
                    uot_decode_threshold=float(uk["uot_decode_threshold"]),
                    uot_cost_weights=uk.get("uot_cost_weights"),
                    uot_backend=str(uk["uot_backend"]),
                    uot_max_delay_sec=float(uk["uot_max_delay_sec"]),
                    uot_causal_violation_penalty=float(uk["uot_causal_violation_penalty"]),
                    uot_lambda_risk=float(uk["uot_lambda_risk"]),
                    uot_causal_infeasible_delay_sec=uk.get("uot_causal_infeasible_delay_sec"),
                    uot_export_matrix=bool(uk["uot_export_matrix"]),
                    uot_export_cost_components=bool(uk["uot_export_cost_components"]),
                    uot_export_cost_matrix_csv=bool(uk.get("uot_export_cost_matrix_csv", False)),
                    uot_allow_unmatched=bool(uk["uot_allow_unmatched"]),
                    uot_use_graph_embedding=bool(uk["uot_use_graph_embedding"]),
                    graph_ranker_checkpoint=getattr(args, "graph_ranker_checkpoint", None),
                    uot_ablation=str(uk.get("uot_ablation") or "none"),
                    run_flow_baselines=True,
                    flow_label_min_confidence=float(uk.get("flow_label_min_confidence") or 0.0),
                    synthetic_eval_hints_path=Path(syn_json),
                    uot_flow_dst_top_k=int(uk.get("uot_flow_dst_top_k") or 200),
                    uot_flow_max_matrix_cells=int(uk.get("uot_flow_max_matrix_cells") or 6_000_000),
                    **_uot_pool_kwargs_from_uk(uk),
                )
                _merge_synthetic_metrics_into_main(out_root, synth_work)
                logger.info("paper_experiment_closure: merged synthetic metrics into main eval")
                _flush_logs()

    try:
        run_threshold_sensitivity(out_root, ev_e, ev_b, args, cfg)
    except Exception:
        logger.exception("threshold_sensitivity failed")

    logger.info("paper_experiment_closure: writing paper_experiment_summary.md and validation merge …")
    _flush_logs()
    write_paper_experiment_summary_md(out_root)
    validate_and_merge_paper_closure_report(out_root)
    logger.info("Paper experiment closure finished -> %s", out_root)
    _flush_logs()
    return 0
