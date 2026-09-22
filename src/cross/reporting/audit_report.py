"""Human-readable post-run audit report (Markdown + JSON) for AML / RC-UOT pipeline outputs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cross.config.output_layout import locate_output_file, output_file


def _resolve_audit_path(root: Path, rel: str) -> Path:
    """``rel`` may be ``paper_tables/foo.csv`` or a flat artifact name."""
    if "/" in rel:
        return root / rel
    return locate_output_file(root, rel)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    if not path.is_file():
        return None, f"missing: {path.name}"
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj, None
    except Exception as e:
        return None, f"unreadable {path.name}: {e}"


def _read_csv(path: Path, *, nrows: int | None = None) -> tuple[pd.DataFrame | None, str | None]:
    if not path.is_file():
        return None, f"missing: {path.name}"
    try:
        df = pd.read_csv(path, nrows=nrows)
        return df, None
    except Exception as e:
        return None, f"unreadable {path.name}: {e}"


def _csv_row_count(path: Path) -> int:
    """Full data row count (excluding header); avoids nrows-limited preview undercounting."""
    if not path.is_file():
        return -1
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            n = sum(1 for _ in f)
        return max(0, n - 1)
    except Exception:
        return -1


def _df_head_md(df: pd.DataFrame, *, max_rows: int) -> str:
    if df is None or df.empty:
        return "_No rows._\n"
    show = df.head(max_rows).copy()
    cols = [str(c) for c in show.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body: list[str] = []
    for _, row in show.iterrows():
        cells: list[str] = []
        for c in show.columns:
            v = row.get(c)
            if v is None:
                s = ""
            elif isinstance(v, float) and pd.isna(v):
                s = ""
            else:
                s = str(v)
            s = s.replace("\n", " ").replace("|", "/")[:200]
            cells.append(s)
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def _blocking_from_token_route(tv: dict[str, Any] | None) -> list[str]:
    if not isinstance(tv, dict):
        return []
    out: list[str] = []
    if tv.get("paper_blocking"):
        out.append("token_route_validation.json: paper_blocking=true")
    msgs = tv.get("blocking_messages") or tv.get("errors") or []
    if isinstance(msgs, list):
        for m in msgs[:12]:
            if isinstance(m, str) and m.strip():
                out.append(f"token_route: {m}")
    return out


def _exec_summary(run_report: dict[str, Any] | None, paper_val: dict[str, Any] | None, out_dir: Path) -> dict[str, Any]:
    rr = run_report if isinstance(run_report, dict) else {}
    pv = paper_val if isinstance(paper_val, dict) else {}
    pbo = rr.get("path_b_options") if isinstance(rr.get("path_b_options"), dict) else {}
    rc = bool(rr.get("rc_uot_executed")) or bool(pbo.get("rc_uot_executed")) or bool(pv.get("rc_uot_executed"))
    paper_ready = bool(rr.get("paper_ready")) if "paper_ready" in rr else bool(pv.get("paper_ready"))
    blocking: list[str] = []
    blocking.extend(_blocking_from_token_route(rr.get("token_route_validation") if isinstance(rr.get("token_route_validation"), dict) else None))
    tr_path = locate_output_file(out_dir, "token_route_validation.json")
    tr_obj, _ = _read_json(tr_path)
    if isinstance(tr_obj, dict):
        blocking.extend(x for x in _blocking_from_token_route(tr_obj) if x not in blocking)
    raw_ps = rr.get("pipeline_status")
    status = str(raw_ps).strip() if raw_ps is not None else ""
    if status and status not in ("completed_ok", "completed_with_evidence_error"):
        blocking.append(f"pipeline_status={status}")
    for w in pv.get("warnings") or []:
        if isinstance(w, str) and "paper_blocking" in w.lower():
            if w not in blocking:
                blocking.append(w)
    run_ok = status in ("completed_ok", "completed_with_evidence_error", "")
    return {
        "timestamp_utc": _utc_now_iso(),
        "output_dir": str(out_dir.resolve()),
        "run_status": status or "not_reported",
        "pipeline_completed_okish": bool(run_ok),
        "main_model": rr.get("main_model"),
        "rc_uot_executed": rc,
        "paper_ready": paper_ready,
        "main_output_level": rr.get("main_output_level"),
        "path_b_matching_method_executed": rr.get("path_b_matching_method_executed"),
        "matching_pairs_csv_role": rr.get("matching_pairs_csv_role"),
        "matching_pairs_note": rr.get("matching_pairs_csv_role") or "see run_report / matching_metrics",
        "blocking_issue_count": len(blocking),
        "blocking_issues": blocking,
    }


def generate_run_audit_report(
    out_dir: str | Path,
    output_markdown: str | Path | None = None,
    output_json: str | Path | None = None,
    max_rows: int = 20,
) -> dict[str, Any]:
    """
    Scan ``out_dir`` for pipeline artifacts and write ``run_audit_report.md`` + ``run_audit_report.json``.

    Missing files are recorded as warnings; generation does not raise on absent artifacts.
    """
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    md_path = Path(output_markdown) if output_markdown else output_file(root, "run_audit_report.md")
    json_path = Path(output_json) if output_json else output_file(root, "run_audit_report.json")
    warnings: list[str] = []
    files_checked: dict[str, str] = {}

    # --- JSON sources ---
    run_report, w = _read_json(_resolve_audit_path(root, "run_report.json"))
    if w:
        warnings.append(w)
    paper_val, w = _read_json(_resolve_audit_path(root, "paper_artifact_validation.json"))
    if w:
        warnings.append(w)
    match_metrics, w = _read_json(_resolve_audit_path(root, "matching_metrics.json"))
    if w:
        warnings.append(w)
    uot_summary, w = _read_json(_resolve_audit_path(root, "uot_summary.json"))
    if w:
        warnings.append(w)
    uot_split, w = _read_json(_resolve_audit_path(root, "uot_split_merge_summary.json"))
    if w:
        warnings.append(w)
    uot_eval, w = _read_json(_resolve_audit_path(root, "uot_evaluation_metrics.json"))
    if w:
        warnings.append(w)
    flow_lvl, w = _read_json(_resolve_audit_path(root, "flow_level_metrics.json"))
    if w:
        warnings.append(w)
    token_route, w = _read_json(_resolve_audit_path(root, "token_route_validation.json"))
    if w:
        warnings.append(w)
    causal_sum, w = _read_json(_resolve_audit_path(root, "causal_feasibility_summary.json"))
    if w:
        warnings.append(w)

    rr = run_report if isinstance(run_report, dict) else {}

    _artifact_index = [
        "run_report.json",
        "paper_artifact_validation.json",
        "matching_metrics.json",
        "uot_summary.json",
        "uot_transport_plan.csv",
        "uot_unmatched_mass.csv",
        "uot_cost_components.csv",
        "uot_cost_matrix.csv",
        "uot_flow_correspondence.csv",
        "uot_split_merge_summary.json",
        "uot_evaluation_metrics.json",
        "flow_level_metrics.json",
        "evidence_eth.csv",
        "evidence_bnb.csv",
        "evidence_candidates.csv",
        "candidate_pool_raw.csv",
        "receipt_verify_debug.csv",
        "uot_flow_segments_eth.csv",
        "uot_flow_segments_bnb.csv",
        "baseline_hungarian.csv",
        "baseline_greedy.csv",
        "pairs.csv",
        "matching_pairs.csv",
        "token_route_validation.json",
        "causal_feasibility_summary.json",
        "traceability_index.csv",
        "uot_marginals.csv",
        "path_b_low_confidence_candidates.csv",
        "ablation_results.csv",
        "paper_tables/table_main_results.csv",
        "paper_tables/table_ablation.csv",
    ]
    for rel in _artifact_index:
        p = _resolve_audit_path(root, rel)
        files_checked[rel] = f"ok ({p.stat().st_size} bytes)" if p.is_file() else "missing"

    # --- CSV previews ---
    def load_csv(rel: str, *, full: bool = False) -> pd.DataFrame | None:
        p = _resolve_audit_path(root, rel)
        cap: int | None = None if full else max_rows * 3
        df, w = _read_csv(p, nrows=cap)
        if w:
            warnings.append(w)
            return None
        return df

    tp = load_csv("uot_transport_plan.csv", full=True)
    um = load_csv("uot_unmatched_mass.csv", full=True)
    ucc = load_csv("uot_cost_components.csv")
    ufc = load_csv("uot_flow_correspondence.csv")
    ev_eth = load_csv("evidence_eth.csv")
    ev_bnb = load_csv("evidence_bnb.csv")
    ev_cand = load_csv("evidence_candidates.csv")
    pool_raw = load_csv("candidate_pool_raw.csv")
    receipt_dbg = load_csv("receipt_verify_debug.csv")
    seg_eth = load_csv("uot_flow_segments_eth.csv")
    seg_bnb = load_csv("uot_flow_segments_bnb.csv")
    seg_eth_a = load_csv("flow_segments_eth.csv")
    seg_bnb_a = load_csv("flow_segments_bnb.csv")
    bl_h = load_csv("baseline_hungarian.csv")
    bl_g = load_csv("baseline_greedy.csv")
    mpairs = load_csv("matching_pairs.csv")
    pairs_a = load_csv("pairs.csv")
    path_a_pairs = load_csv("path_a_pairs.csv")
    ablation = load_csv("ablation_results.csv")
    trace = load_csv("traceability_index.csv")
    marg = load_csv("uot_marginals.csv")
    ucm = load_csv("uot_cost_matrix.csv")
    low_conf = load_csv("path_b_low_confidence_candidates.csv")

    # --- integrity counts ---
    def nrows(df: pd.DataFrame | None) -> int:
        if df is None:
            return -1
        return int(len(df))

    integrity = {
        "uot_transport_plan_rows": _csv_row_count(_resolve_audit_path(root, "uot_transport_plan.csv")),
        "uot_unmatched_mass_rows": _csv_row_count(_resolve_audit_path(root, "uot_unmatched_mass.csv")),
        "uot_cost_components_rows": _csv_row_count(_resolve_audit_path(root, "uot_cost_components.csv")),
        "uot_flow_correspondence_rows": _csv_row_count(_resolve_audit_path(root, "uot_flow_correspondence.csv")),
        "evidence_eth_rows": _csv_row_count(_resolve_audit_path(root, "evidence_eth.csv")),
        "evidence_bnb_rows": _csv_row_count(_resolve_audit_path(root, "evidence_bnb.csv")),
        "evidence_candidates_rows": _csv_row_count(_resolve_audit_path(root, "evidence_candidates.csv")),
        "uot_flow_segments_eth_rows": _csv_row_count(_resolve_audit_path(root, "uot_flow_segments_eth.csv")),
        "uot_flow_segments_bnb_rows": _csv_row_count(_resolve_audit_path(root, "uot_flow_segments_bnb.csv")),
    }

    # --- top transport mass ---
    top_transport: list[dict[str, Any]] = []
    if tp is not None and not tp.empty:
        mass_col = "transport_mass" if "transport_mass" in tp.columns else None
        if mass_col:
            t2 = tp.copy()
            t2["_m"] = pd.to_numeric(t2[mass_col], errors="coerce").fillna(0.0)
            top_transport = (
                t2.nlargest(min(max_rows, len(t2)), "_m")
                .drop(columns=["_m"], errors="ignore")
                .to_dict(orient="records")
            )
        else:
            top_transport = tp.head(max_rows).to_dict(orient="records")

    # --- unmatched highlights ---
    unmatched_high: list[dict[str, Any]] = []
    if um is not None and not um.empty:
        for col in ("unmatched_mass", "unmatched_ratio", "original_mass"):
            if col in um.columns:
                u2 = um.copy()
                u2["_v"] = pd.to_numeric(u2[col], errors="coerce").fillna(0.0).abs()
                unmatched_high = (
                    u2.nlargest(min(max_rows, len(u2)), "_v")
                    .drop(columns=["_v"], errors="ignore")
                    .to_dict(orient="records")
                )
                break
        if not unmatched_high:
            unmatched_high = um.head(max_rows).to_dict(orient="records")

    # --- low confidence / weak evidence ---
    low_rows: list[dict[str, Any]] = []
    if low_conf is not None and not low_conf.empty:
        low_rows = low_conf.head(max_rows).to_dict(orient="records")
    elif ev_cand is not None and not ev_cand.empty and "candidate_status" in ev_cand.columns:
        sub = ev_cand[ev_cand["candidate_status"].astype(str).str.lower().isin(("low_confidence", "invalid"))]
        low_rows = sub.head(max_rows).to_dict(orient="records")
    elif ev_cand is not None and not ev_cand.empty and "edge_role" in ev_cand.columns:
        sub = ev_cand[ev_cand["edge_role"].astype(str).str.contains("raw", case=False, na=False)]
        low_rows = sub.head(max_rows).to_dict(orient="records")

    # --- baseline vs RC-UOT (lightweight) ---
    baseline_vs: dict[str, Any] = {
        "baseline_hungarian_rows": _csv_row_count(_resolve_audit_path(root, "baseline_hungarian.csv")),
        "baseline_greedy_rows": _csv_row_count(_resolve_audit_path(root, "baseline_greedy.csv")),
        "matching_pairs_rows": _csv_row_count(_resolve_audit_path(root, "matching_pairs.csv")),
        "pairs_csv_rows": _csv_row_count(_resolve_audit_path(root, "pairs.csv")),
        "path_a_pairs_rows": _csv_row_count(_resolve_audit_path(root, "path_a_pairs.csv")),
        "note": "Compare matching_pairs / baselines to uot_transport_plan for tx-level vs flow-level semantics.",
    }
    if mpairs is not None and bl_h is not None and not mpairs.empty and not bl_h.empty:
        sh = "srcTxHash" if "srcTxHash" in mpairs.columns else "src_tx_hash" if "src_tx_hash" in mpairs.columns else None
        dh = "dstTxHash" if "dstTxHash" in mpairs.columns else "dst_tx_hash" if "dst_tx_hash" in mpairs.columns else None
        sh2 = "srcTxHash" if "srcTxHash" in bl_h.columns else None
        dh2 = "dstTxHash" if "dstTxHash" in bl_h.columns else None
        if sh and dh and sh2 and dh2:
            mset = set(zip(mpairs[sh].astype(str).str.lower(), mpairs[dh].astype(str).str.lower()))
            hset = set(zip(bl_h[sh2].astype(str).str.lower(), bl_h[dh2].astype(str).str.lower()))
            baseline_vs["pairs_vs_hungarian_agreement_frac"] = round(
                len(mset & hset) / max(len(mset), 1),
                4,
            )

    exec_sum = _exec_summary(rr, paper_val if isinstance(paper_val, dict) else None, root)

    next_steps: list[str] = []
    if not _resolve_audit_path(root, "run_report.json").is_file():
        next_steps.append("Open run log; ensure pipeline wrote run_report.json.")
    if exec_sum["blocking_issue_count"]:
        next_steps.append("Review token_route_validation.json and pipeline_status in run_report.json.")
    if not exec_sum["paper_ready"]:
        next_steps.append("Read paper_artifact_validation.json for missing_files / empty_files / required_columns_check.")
    if nrows(tp) <= 0:
        next_steps.append("Inspect Path B logs and uot_diagnostics.json; confirm flow segments and UOT solver ran.")
    if nrows(ev_cand) <= 0 and nrows(ev_eth) <= 0:
        next_steps.append("Check evidence_export_status in matching_metrics.json and evidence_export_debug.json if present.")
    next_steps.append("For quantitative review: uot_transport_plan.csv, uot_unmatched_mass.csv, uot_evaluation_metrics.json.")
    next_steps.append("For receipt audit: receipt_verify_debug.csv and evidence_candidates.csv.")

    result: dict[str, Any] = {
        "executive_summary": exec_sum,
        "files_checked": files_checked,
        "warnings": warnings,
        "integrity_row_counts": integrity,
        "uot_summary": uot_summary if isinstance(uot_summary, dict) else {},
        "uot_split_merge_summary": uot_split if isinstance(uot_split, dict) else {},
        "uot_evaluation_metrics": uot_eval if isinstance(uot_eval, dict) else {},
        "flow_level_metrics": flow_lvl if isinstance(flow_lvl, dict) else {},
        "matching_metrics_excerpt": (
            {
                k: match_metrics[k]
                for k in ("mode", "matching_pairs_csv_role", "accuracy", "evidence_export_status")
                if k in match_metrics
            }
            if isinstance(match_metrics, dict)
            else {}
        ),
        "paper_artifact_validation": paper_val if isinstance(paper_val, dict) else {},
        "token_route_validation": token_route if isinstance(token_route, dict) else {},
        "causal_feasibility_summary": causal_sum if isinstance(causal_sum, dict) else {},
        "top_transport_edges": top_transport,
        "unmatched_mass_highlights": unmatched_high,
        "low_confidence_or_weak_candidates": low_rows,
        "baseline_vs_uot": baseline_vs,
        "label_usage": rr.get("label_usage"),
        "next_recommended_files": next_steps,
    }

    # --- Markdown ---
    lines: list[str] = []
    lines.append("# Run Audit Report\n")
    lines.append("## 1. Executive Summary\n")
    lines.append("Core conclusions for this run (human scan).\n")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Timestamp (UTC) | `{exec_sum['timestamp_utc']}` |")
    lines.append(f"| Output directory | `{exec_sum['output_dir']}` |")
    lines.append(f"| Run / pipeline status | `{exec_sum['run_status']}` |")
    lines.append(f"| Main model | `{exec_sum.get('main_model')}` |")
    lines.append(f"| RC-UOT executed | `{exec_sum.get('rc_uot_executed')}` |")
    lines.append(f"| Paper ready | `{exec_sum.get('paper_ready')}` |")
    lines.append(f"| Main output level | `{exec_sum.get('main_output_level')}` |")
    lines.append(f"| Path B matching executed | `{exec_sum.get('path_b_matching_method_executed')}` |")
    lines.append(f"| Matching pairs role | `{exec_sum.get('matching_pairs_csv_role') or 'see run_report / matching_metrics'}` |")
    lines.append(f"| Blocking issues | `{exec_sum['blocking_issue_count']}` |")
    lines.append("")

    if exec_sum["blocking_issues"]:
        lines.append("**Blocking issues**\n")
        for b in exec_sum["blocking_issues"]:
            lines.append(f"- {b}")
        lines.append("")

    lines.append("## 2. Run success and artifacts\n")
    lines.append(f"- Pipeline completed (soft): **{exec_sum['pipeline_completed_okish']}**\n")
    if warnings:
        lines.append("### Warnings (missing or unreadable files)\n")
        for w in warnings[:40]:
            lines.append(f"- {w}")
        if len(warnings) > 40:
            lines.append(f"- _… and {len(warnings) - 40} more (see JSON)._")
        lines.append("")
    lines.append("### Row / size sanity\n")
    lines.append("| Artifact | Rows or note |")
    lines.append("|---|---|")
    for k, v in integrity.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## 3. RC-UOT and paper readiness\n")
    lines.append("### uot_summary.json (excerpt)\n")
    if isinstance(uot_summary, dict) and uot_summary:
        for key in ("n_eth_flows", "n_bnb_flows", "row_entropy_avg", "decode_threshold", "backend"):
            if key in uot_summary:
                lines.append(f"- **{key}**: `{uot_summary.get(key)}`")
        lines.append("")
    else:
        lines.append("_Not present or empty._\n")

    lines.append("### paper_artifact_validation.json\n")
    if isinstance(paper_val, dict) and paper_val:
        lines.append(f"- **paper_ready**: `{paper_val.get('paper_ready')}`")
        lines.append(f"- **rc_uot_executed**: `{paper_val.get('rc_uot_executed')}`")
        lines.append(f"- **main_model_ok**: `{paper_val.get('main_model_ok')}`")
        mf = paper_val.get("missing_files") or []
        ef = paper_val.get("empty_files") or []
        if mf:
            lines.append(f"- **missing_files** ({len(mf)}): {', '.join(str(x) for x in mf[:15])}")
        if ef:
            lines.append(f"- **empty_files** ({len(ef)}): {', '.join(str(x) for x in ef[:15])}")
        lines.append("")
    else:
        lines.append("_Not present._\n")

    lines.append("## 4. Evidence layer and flow segments\n")
    lines.append("### Evidence CSV row counts (full file counts where available)\n")
    lines.append(f"- evidence_eth: {integrity.get('evidence_eth_rows', nrows(ev_eth))}")
    lines.append(f"- evidence_bnb: {integrity.get('evidence_bnb_rows', nrows(ev_bnb))}")
    lines.append(f"- evidence_candidates: {integrity.get('evidence_candidates_rows', nrows(ev_cand))}")
    lines.append(f"- candidate_pool_raw: {_csv_row_count(_resolve_audit_path(root, 'candidate_pool_raw.csv'))}")
    lines.append(f"- receipt_verify_debug: {nrows(receipt_dbg)}")
    lines.append("")
    lines.append("### Flow segment files\n")
    lines.append(
        f"- uot_flow_segments_eth: {integrity.get('uot_flow_segments_eth_rows', nrows(seg_eth))}; "
        f"uot_flow_segments_bnb: {integrity.get('uot_flow_segments_bnb_rows', nrows(seg_bnb))}"
    )
    lines.append(
        f"- flow_segments_eth (alias): {_csv_row_count(_resolve_audit_path(root, 'flow_segments_eth.csv'))}; "
        f"flow_segments_bnb: {_csv_row_count(_resolve_audit_path(root, 'flow_segments_bnb.csv'))}"
    )
    lines.append("")

    lines.append("## 5. UOT transport plan (most confident mass)\n")
    lines.append(f"Top edges by transport mass (up to {max_rows} rows):\n")
    if tp is not None and not tp.empty:
        mass_col = "transport_mass" if "transport_mass" in tp.columns else None
        if mass_col:
            tshow = tp.copy()
            tshow["_sort"] = pd.to_numeric(tshow[mass_col], errors="coerce").fillna(0.0)
            tshow = tshow.nlargest(min(max_rows, len(tshow)), "_sort").drop(columns=["_sort"], errors="ignore")
        else:
            tshow = tp.head(max_rows)
        lines.append(_df_head_md(tshow, max_rows=max_rows))
    else:
        lines.append("_No transport plan loaded._\n")

    lines.append("## 6. Unmatched mass (high residual)\n")
    if unmatched_high:
        lines.append(_df_head_md(pd.DataFrame(unmatched_high), max_rows=max_rows))
    else:
        lines.append("_No unmatched mass table or empty._\n")

    lines.append("## 7. Low confidence / weak evidence candidates\n")
    if low_rows:
        lines.append(_df_head_md(pd.DataFrame(low_rows), max_rows=max_rows))
    else:
        lines.append("_No dedicated low-confidence table; none flagged in evidence_candidates._\n")

    lines.append("## 8. Baselines vs RC-UOT\n")
    lines.append(f"- baseline_hungarian rows: {baseline_vs['baseline_hungarian_rows']}")
    lines.append(f"- baseline_greedy rows: {baseline_vs['baseline_greedy_rows']}")
    lines.append(f"- matching_pairs rows: {baseline_vs['matching_pairs_rows']}")
    if "pairs_vs_hungarian_agreement_frac" in baseline_vs:
        lines.append(f"- decoded pairs vs Hungarian overlap (rough): **{baseline_vs['pairs_vs_hungarian_agreement_frac']}**")
    lines.append(f"- _{baseline_vs.get('note', '')}_\n")

    lines.append("## 9. Blocking issues, validation, ablation, traceability\n")
    lines.append("### Token route & causal checks\n")
    if isinstance(token_route, dict) and token_route:
        lines.append(f"- token_route **paper_blocking**: `{token_route.get('paper_blocking')}`")
    else:
        lines.append("- token_route_validation.json: _missing or empty_")
    if isinstance(causal_sum, dict) and causal_sum:
        lines.append(f"- causal_feasibility_summary keys: `{', '.join(list(causal_sum.keys())[:12])}`")
    else:
        lines.append("- causal_feasibility_summary.json: _missing_")
    lines.append("")
    lines.append("### Ablation & paper tables (optional)\n")
    if ablation is not None and not ablation.empty:
        lines.append(_df_head_md(ablation, max_rows=min(10, max_rows)))
    else:
        lines.append("_ablation_results.csv missing or empty._\n")
    for rel in ("paper_tables/table_main_results.csv", "paper_tables/table_ablation.csv"):
        p = root / rel
        if p.is_file():
            tdf, _ = _read_csv(p, nrows=15)
            lines.append(f"#### `{rel}`\n")
            lines.append(_df_head_md(tdf, max_rows=15) if tdf is not None else "_Unreadable._\n")
        else:
            lines.append(f"- _Missing_: `{rel}`")
    lines.append("")
    lines.append("### Traceability / marginals (row counts)\n")
    lines.append(f"- traceability_index: {nrows(trace)}; uot_marginals: {nrows(marg)}; uot_cost_matrix: {nrows(ucm)}")
    lines.append("")

    lines.append("## 10. Next files to inspect\n")
    for s in next_steps:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("---\n")
    lines.append("*Generated by `cross.reporting.audit_report.generate_run_audit_report`.*\n")

    md_text = "\n".join(lines)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text, encoding="utf-8")
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return result
