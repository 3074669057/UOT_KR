"""Paper finalization: artifact audit, decode interpretation, and paper-facing summaries."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from cross.config.output_layout import output_file
from cross.domain.evaluation.decode_threshold_sweep import (
    RECOMMENDED_DECODE_SELECTION_DOC,
    recommend_decode_rule_from_sweep_dataframe,
)

logger = logging.getLogger(__name__)

_TOP_N = 15
_REQUIRED_SWEEP_COLS = (
    "source_share_ge",
    "topk_per_source",
    "cumulative_row_mass",
    "decode_rule",
    "threshold_or_k",
    "num_predicted_edges",
    "num_true_edges",
    "num_true_positive_edges",
    "flow_pair_precision",
    "flow_pair_recall",
    "flow_pair_f1",
    "flow_mass_recall",
    "flow_mass_precision",
    "top1_flow_accuracy",
    "top3_flow_accuracy",
    "average_edges_per_source",
)
_METRIC_COLS = (
    "flow_pair_precision",
    "flow_pair_recall",
    "flow_pair_f1",
    "flow_mass_recall",
    "flow_mass_precision",
    "top1_flow_accuracy",
    "top3_flow_accuracy",
    "average_edges_per_source",
    "num_predicted_edges",
    "num_true_edges",
    "num_true_positive_edges",
)


def _float_close(a: Any, b: Any) -> bool:
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    if math.isnan(x) and math.isnan(y):
        return True
    return math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-6)


def _recommend_matches_file(computed: dict[str, Any], on_disk: dict[str, Any]) -> tuple[bool, list[str]]:
    diffs: list[str] = []
    keys = (
        "recommended_rule",
        "recommended_threshold_or_k",
        "flow_pair_f1",
        "flow_pair_recall",
        "flow_mass_recall",
        "average_edges_per_source",
        "source_share_ge",
        "topk_per_source",
        "cumulative_row_mass",
    )
    for k in keys:
        c_v, d_v = computed.get(k), on_disk.get(k)
        if k in ("recommended_rule", "recommended_threshold_or_k"):
            if str(c_v) != str(d_v):
                diffs.append(f"{k}: computed={c_v!r} file={d_v!r}")
            continue
        if c_v is None and d_v is None:
            continue
        if not _float_close(c_v, d_v):
            diffs.append(f"{k}: computed={c_v!r} file={d_v!r}")
    return (len(diffs) == 0, diffs)


def _nan_report(df: pd.DataFrame, cols: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in cols:
        if c not in df.columns:
            out[c] = "missing_column"
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        n_na = int(s.isna().sum())
        out[c] = {"na_count": n_na, "non_numeric_coerced": n_na}
    return out


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x: Any, nd: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return "nan"
    return f"{v:.{nd}f}"


def _write_paper_results_decode_manuscript(
    exp: Path,
    *,
    effective_rec: dict[str, Any] | None,
    rec_row: pd.Series | None,
    topk5_row: pd.Series | None,
) -> None:
    """Manuscript-oriented decode paragraph (real Celer weak labels; fixed dense ``P``)."""
    lines: list[str] = [
        "# Paper results: decode-rule sensitivity (real Celer weak labels)",
        "",
        "**Population and inputs:** weak ``flow_labels.csv`` against the **main real-data** dense transport matrix "
        "``uot/uot_transport_matrix.npz`` (no re-solve). This is **not** the semi-synthetic workspace and **not** an oracle pool.",
        "",
        "## Recommended decode and exact metrics",
        "",
    ]
    if effective_rec:
        thr = str(effective_rec.get("recommended_threshold_or_k") or "")
        lines.append(
            f"The grid-selected rule is **{effective_rec.get('recommended_rule')}** with parameters "
            f"**`{thr}`**. On this snapshot it achieves **flow_pair_f1 = {_fmt(effective_rec.get('flow_pair_f1'))}**, "
            f"**flow_pair_recall = {_fmt(effective_rec.get('flow_pair_recall'))}**, "
            f"**flow_mass_recall = {_fmt(effective_rec.get('flow_mass_recall'))}** (USD-aligned mass on true pairs "
            "conditional on the decoded plan), and **average_edges_per_source = "
            f"{_fmt(effective_rec.get('average_edges_per_source'), 3)}**."
        )
        if rec_row is not None:
            if "flow_pair_precision" in rec_row.index:
                lines.append(
                    f"**Edge precision** on the same row is **{_fmt(rec_row.get('flow_pair_precision'))}**; "
                    f"**flow_mass_precision = {_fmt(rec_row.get('flow_mass_precision'))}** when available."
                )
        lines.append("")
    else:
        lines.extend(["_No `recommended_decode_rule.json` / sweep row available._", ""])

    lines.extend(
        [
            "## Why ``flow_pair_f1`` is the primary selector",
            "",
            "The sweep ranks combined decodes by **``flow_pair_f1``** because the paper question at this stage is "
            "how to turn a **soft** transport matrix into a **sparse set of reported edges** without hand-tuning a single "
            "mass cut: F1 directly summarizes the tradeoff between **edge precision** and **edge recall** on weak labels. "
            "Tie-breakers prefer **``flow_mass_recall``** then **``flow_pair_recall``** so that among near-tied F1 rows we "
            "favor plans that keep more labeled source USD on true pairs and recover more true edges.",
            "",
            "## Why ``flow_mass_recall`` must appear next to edge-level F1",
            "",
            "RC-UOT can keep **high mass on true pairs** while spreading probability across many admissible cells. "
            "**``flow_mass_recall``** (as implemented in the sweep) measures USD-weighted mass attributed to true pairs "
            "**relative to the decoded multi-edge plan**, whereas **``flow_pair_f1``** only credits **discrete** edges that "
            "survive the conjunction. A manuscript should therefore **pair** edge F1 with mass recall: a high mass recall "
            "with modest F1 signals **diffuse** transport under the same decode, not necessarily poor matching on mass.",
            "",
            "## ``topk = 1`` (recommended) vs high-recall ``topk = 5``",
            "",
        ]
    )
    if topk5_row is not None and effective_rec:
        lines.append(
            f"**Recommended row ({thr})** keeps **at most one** high-share edge per source after intersecting with "
            f"share and cumulative-mass masks, yielding **~{_fmt(effective_rec.get('average_edges_per_source'), 3)}** edges per source on average "
            f"and the F1 / recall pair above. By contrast, a **high-recall ``topk = 5``** representative from the same sweep "
            f"(ranking ``topk=5`` rows by ``flow_pair_recall``) gives **F1 = {_fmt(topk5_row.get('flow_pair_f1'))}**, "
            f"**recall = {_fmt(topk5_row.get('flow_pair_recall'))}**, **mass recall = {_fmt(topk5_row.get('flow_mass_recall'))}**, "
            f"and **~{_fmt(topk5_row.get('average_edges_per_source'), 3)}** edges per source — recovering more weak-label edges "
            "but at **lower precision** and a **denser** decoded graph."
        )
        lines.append("")
    else:
        lines.append(
            "_Insufficient sweep rows to contrast ``topk=1`` vs ``topk=5``; see ``decode_threshold_tradeoff.md`` once the "
            "54-row grid is present._"
        )
        lines.append("")

    lines.extend(
        [
            "## Soft transport mass vs hard decoded edges",
            "",
            "The dense matrix ``P`` encodes **soft** coupling: many cells can carry small positive mass. The conjunction "
            "``(share_ge ∧ topk ∧ cumulative_row_mass)`` produces a **hard** edge list for evaluation. **Soft mass** can "
            "remain on pruned cells even when **hard** F1 is high; conversely, permissive decodes can increase **hard** "
            "edges (higher recall, lower precision) while **mass recall** moves only modestly if mass was already concentrated. "
            "Treat **mass metrics** as accounting for **where** probability sits and **pair-F1** as summarizing **which** discrete edges you report.",
            "",
            "## Artifacts",
            "",
            "- ``experiments/decode_threshold_sweep.csv`` — full 6×3×3 combined grid",
            "- ``experiments/recommended_decode_rule.json`` — programmatic best-F1 row",
            "- ``experiments/decode_threshold_tradeoff.md`` — tables and lever discussion",
            "- ``experiments/decode_threshold_top_rules.csv`` — top rows by F1",
            "",
        ]
    )
    (exp / "paper_results_decode.md").write_text("\n".join(lines), encoding="utf-8")


def _write_synthetic_metric_interpretation_md(out_root: Path) -> None:
    """Document how semi-synthetic scenario CSV metrics are defined in code."""
    p = out_root / "experiments" / "synthetic_metric_interpretation.md"
    body = """# Semi-synthetic scenario metrics (definitions and reading the CSV)

This note matches the implementation in ``src/cross/domain/evaluation/synthetic_scenario_eval.py`` (per-scenario CSV rows).
The global flow-eval field ``unmatched_mass_detection_f1`` uses a related construction in ``src/cross/domain/evaluation/flow_eval.py``
(``_unmatched_detection_f1``) when reporting full-run metrics; the **scenario table** for ``unmatched`` follows the same **0.08**
``unmatched_ratio`` gate but is computed in ``write_synthetic_eval_by_scenario``.

## ``unmatched_detection_f1`` (``unmatched`` scenario row)

**Definition (scenario CSV):** among ETH rows in ``uot_unmatched_mass.csv``, any source whose ``unmatched_ratio >= 0.08``
is treated as a **positive prediction** that the flow is “highly unmatched”. The set ``truth_unmatched_src_flows`` from
synthetic eval hints is the **ground-truth** set of synthetic unmatched sources. Standard set-based **precision / recall / F1**
is computed between those two sets.

**Reading ``0.0``:** this is a real **F1 score**, not a sentinel for “missing metric”. It means there was **no overlap**
between truth unmatched sources and sources flagged by the ``>= 0.08`` rule (typically **true positives = 0** with false
negatives on the truth set). In our runs this usually tracks **near-zero ``unmatched_ratio``** in USD-normalized columns for
synthetic unmatched rows, so the fixed **0.08** gate never fires — an **evaluation-threshold / normalization mismatch**, not
necessarily a claim that RC-UOT “cannot” assign unmatched mass. See ``eval/synthetic_failure_debug.csv`` for row-level notes.

**Code change vs prose:** default behavior is **intentional** given the current definition; improving the story is primarily
**prose and/or scenario-specific thresholds**, unless you explicitly change ``um_ratio_thr`` or the synthetic unmatched mass
construction.

## ``decoy_pair_match_rate`` (``delay_noise`` scenario row)

**Definition:** let ``noise_decoy_pairs`` be the list of decoy ``(src_flow_id, dst_flow_id)`` edges from synthetic eval hints.
Let ``pred`` be the set of edges with **positive transport mass** in ``uot_transport_plan.csv`` (above a tiny mass floor).
Then:

``decoy_pair_match_rate = (# of decoy pairs that appear in pred) / (total # of decoy pairs)``.

**Reading ``1.0``:** **every** listed decoy edge carries positive mass in the exported plan — i.e. **all decoys are “hit” by
the soft plan**. That is **bad** for decoy rejection in plain language. The name is **literal** (it is a *match / hit rate*,
not a “success rate” for rejection).

Downstream bundling maps this to ``synthetic_decoy_rejection_rate = 1 - decoy_pair_match_rate`` in
``build_synthetic_metrics_bundle`` — so a CSV value of **1.0** corresponds to **0** rejection in that derived field.

**Code change vs prose:** the metric is **not** counterintuitively inverted in code; clarify in text that “match” means
“appears in the decoded/positive-mass edge set”. Optional renames are a documentation / API polish choice, not required to
interpret the current number.
"""
    p.write_text(body, encoding="utf-8")


def _sweep_row(out_root: Path, strategy: str) -> dict[str, str]:
    p = out_root / "experiments" / "candidate_pool_sweep.csv"
    if not p.is_file():
        return {}
    try:
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        sub = df[df["strategy"].astype(str) == strategy]
        if sub.empty:
            return {}
        r = sub.iloc[0]
        return {k: str(r[k]) for k in ("candidate_dst_recall", "bnb_active", "matrix_cells") if k in r.index}
    except Exception:
        return {}


def run_paper_finalization(out_root: Path, *, top_n: int = _TOP_N) -> int:
    """
    Write audit + interpretation + paper-facing markdown under ``out_root/experiments/``.

    Returns 0 if all audit checks pass, else 1.
    """
    out_root = Path(out_root)
    exp = out_root / "experiments"
    exp.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    all_passed = True

    sweep_p = exp / "decode_threshold_sweep.csv"
    rec_p = exp / "recommended_decode_rule.json"
    lbs_p = exp / "large_budget_setting.json"
    summary_p = output_file(out_root, "paper_experiment_summary.md")

    # --- 1) Audit sweep CSV ---
    row_count = -1
    df = pd.DataFrame()
    if sweep_p.is_file():
        df = pd.read_csv(sweep_p, dtype=str, keep_default_na=False)
        row_count = len(df)
    ok_rows = row_count == 54
    checks.append({"name": "decode_threshold_sweep_rows", "expected": 54, "actual": row_count, "pass": ok_rows})
    all_passed = all_passed and ok_rows

    missing_cols = [c for c in _REQUIRED_SWEEP_COLS if c not in df.columns]
    ok_cols = len(missing_cols) == 0
    checks.append({"name": "decode_threshold_sweep_columns", "missing": missing_cols, "pass": ok_cols})
    all_passed = all_passed and ok_cols

    nan_rep: dict[str, Any] = {}
    malformed = False
    if ok_cols and not df.empty:
        df_num = df.copy()
        for c in _METRIC_COLS:
            if c in df_num.columns:
                df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
        nan_rep = _nan_report(df_num, _METRIC_COLS)
        for c, info in nan_rep.items():
            if isinstance(info, dict) and info.get("na_count", 0) > 0:
                malformed = True
        checks.append(
            {
                "name": "decode_threshold_sweep_nan_scan",
                "detail": nan_rep,
                "pass": not malformed,
            }
        )
        all_passed = all_passed and (not malformed)

    # --- recommended JSON vs CSV ---
    rec_ok = False
    rec_diffs: list[str] = []
    computed_rec: dict[str, Any] | None = None
    on_disk_rec: dict[str, Any] | None = None
    if rec_p.is_file():
        try:
            on_disk_rec = _read_json(rec_p)
        except Exception as e:
            rec_diffs = [f"recommended_json_read: {e}"]
            on_disk_rec = None
    if not df.empty and on_disk_rec is not None:
        try:
            df_f = df.copy()
            for c in ("flow_pair_f1", "flow_mass_recall", "flow_pair_recall", "average_edges_per_source"):
                if c in df_f.columns:
                    df_f[c] = pd.to_numeric(df_f[c], errors="coerce")
            computed_rec = recommend_decode_rule_from_sweep_dataframe(df_f)
            if computed_rec and on_disk_rec:
                rec_ok, rec_diffs = _recommend_matches_file(computed_rec, on_disk_rec)
        except Exception as e:
            rec_diffs = [f"exception: {e}"]
            rec_ok = False
    elif not rec_p.is_file():
        rec_ok = False
        rec_diffs = ["missing experiments/recommended_decode_rule.json"]
    effective_rec: dict[str, Any] | None = computed_rec or on_disk_rec
    checks.append(
        {
            "name": "recommended_decode_rule_matches_csv",
            "selection_rule": RECOMMENDED_DECODE_SELECTION_DOC,
            "pass": rec_ok,
            "diffs": rec_diffs,
        }
    )
    all_passed = all_passed and rec_ok

    ok_lbs = lbs_p.is_file()
    checks.append({"name": "large_budget_setting_json", "path": str(lbs_p), "pass": ok_lbs})
    all_passed = all_passed and ok_lbs

    ok_summary = summary_p.is_file()
    checks.append({"name": "paper_experiment_summary_md", "path": str(summary_p), "pass": ok_summary})
    all_passed = all_passed and ok_summary

    manifest = {
        "out_root": str(out_root.resolve()),
        "all_audit_checks_passed": all_passed,
        "recommended_decode_selection": RECOMMENDED_DECODE_SELECTION_DOC,
        "checks": checks,
        "nan_and_coercion_scan": nan_rep,
    }
    (exp / "final_artifact_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # --- Markdown audit ---
    audit_lines = [
        "# Final artifact audit",
        "",
        f"- **Output root:** `{out_root.resolve()}`",
        f"- **Overall:** {'**PASS**' if all_passed else '**FAIL**'}",
        "",
        "## Recommended-rule selection (for verification)",
        "",
        RECOMMENDED_DECODE_SELECTION_DOC,
        "",
        "## Checks",
        "",
    ]
    for ch in checks:
        audit_lines.append(f"- **{ch['name']}:** {'PASS' if ch.get('pass') else 'FAIL'} — `{json.dumps(ch, ensure_ascii=False)}`")
    audit_lines.extend(["", "## NaN / coercion scan (metric columns)", "", "```json", json.dumps(nan_rep, indent=2), "```", ""])
    (exp / "final_artifact_audit.md").write_text("\n".join(audit_lines), encoding="utf-8")
    _write_synthetic_metric_interpretation_md(out_root)

    # --- 2) Decode interpretation (needs valid df) ---
    if not df.empty and "flow_pair_f1" in df.columns:
        dfx = df.copy()
        for c in _METRIC_COLS:
            if c in dfx.columns:
                dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
        sub = dfx[dfx["decode_rule"].astype(str) == "combined_share_topk_cumulative"].copy()
        top_f1 = sub.sort_values("flow_pair_f1", ascending=False, na_position="last").head(top_n)
        top_f1.to_csv(exp / "decode_threshold_top_rules.csv", index=False)

        top_recall = sub.sort_values(
            ["flow_pair_recall", "flow_pair_f1", "flow_pair_precision"],
            ascending=[False, False, False],
            na_position="last",
        ).head(top_n)

        trade_lines = [
            "# Decode threshold tradeoff interpretation",
            "",
            "## Scope",
            "",
            "**Real Celer weak-label setting:** metrics are computed from the fixed dense transport plan "
            "``uot/uot_transport_matrix.npz`` and ``flow_labels.csv`` (no UOT re-solve). "
            "This is **not** semi-synthetic stress data and **not** an oracle pool diagnostic.",
            "",
            "## Recommended rule (documented selection)",
            "",
            RECOMMENDED_DECODE_SELECTION_DOC,
            "",
        ]
        if effective_rec:
            trade_lines.extend(
                [
                    "Current file-backed recommendation (matches CSV programmatically when audit passes):",
                    "",
                    f"- **Rule:** `{effective_rec.get('recommended_rule')}`",
                    f"- **Thresholds:** `{effective_rec.get('recommended_threshold_or_k')}`",
                    f"- **flow_pair_f1:** {effective_rec.get('flow_pair_f1')}",
                    f"- **flow_pair_recall / flow_mass_recall:** {effective_rec.get('flow_pair_recall')} / {effective_rec.get('flow_mass_recall')}",
                    f"- **average_edges_per_source:** {effective_rec.get('average_edges_per_source')}",
                    "",
                    "**Why this row wins:** it maximizes edge-level F1 on the grid; ties prefer higher "
                    "``flow_mass_recall`` then ``flow_pair_recall``, which keeps more labeled USD mass on "
                    "true pairs *conditional on the decoded edge set* while favoring more recovered edges.",
                    "",
                ]
            )

        trade_lines.extend(
            [
                "## Top rules by ``flow_pair_f1`` (see ``decode_threshold_top_rules.csv``)",
                "",
                "The CSV lists the top ``%d`` combined-grid rows sorted by F1." % top_n,
                "",
                "## Secondary view: emphasize recall",
                "",
                "Rows sorted primarily by ``flow_pair_recall`` (then F1, then precision) are previewed below "
                "(full top-N overlap with CSV columns).",
                "",
            ]
        )
        trade_lines.append("| rank | share_ge | topk | cum_mass | F1 | precision | recall | mass_recall | avg_edges |")
        trade_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for i, (_, r) in enumerate(top_recall.iterrows(), 1):
            pr = r.get("flow_pair_precision", "")
            if pr == "" or (isinstance(pr, float) and math.isnan(float(pr))):
                pr = "n/a"
            trade_lines.append(
                f"| {i} | {r.get('source_share_ge', '')} | {r.get('topk_per_source', '')} | {r.get('cumulative_row_mass', '')} "
                f"| {r.get('flow_pair_f1', '')} | {pr} | {r.get('flow_pair_recall', '')} "
                f"| {r.get('flow_mass_recall', '')} | {r.get('average_edges_per_source', '')} |"
            )

        trade_lines.extend(
            [
                "",
                "## Qualitative levers",
                "",
                "### (a) ``source_share_ge``",
                "Higher values prune low–source-share cells; sparser edges, often higher precision, lower recall.",
                "",
                "### (b) ``topk_per_source``",
                "Caps how many destinations per source survive the conjunction; low **k** forces a near one-edge-per-source decode.",
                "",
                "### (c) ``cumulative_row_mass``",
                "Higher fractions keep more mass-bearing columns per row before intersecting with share/top-k; "
                "lower fractions aggressively thin edges.",
                "",
                "## Edge F1 vs soft transport mass",
                "",
                "**``flow_pair_f1``** rewards discrete edges that match weak labels. **``flow_mass_recall``** (USD proxy) "
                "can stay high when mass remains diffuse across many admissible pairs: tightening the conjunction "
                "often **cuts** F1 if it removes true pairs, but can **preserve** mass recall if remaining mass is "
                "re-routed onto other high-mass cells. Conversely, permissive decoding increases recall/F1 on edges "
                "but may add false edges and lower precision. **Report both** when discussing RC-UOT decoding.",
                "",
                "## Figure",
                "",
                "_No matplotlib/seaborn dependency in this repository; no auto-generated figure. "
                "Export CSV and plot externally for publication._",
                "",
            ]
        )
        (exp / "decode_threshold_tradeoff.md").write_text("\n".join(trade_lines), encoding="utf-8")

        rec_row: pd.Series | None = None
        thr_key = str((effective_rec or {}).get("recommended_threshold_or_k") or "")
        if thr_key and not sub.empty:
            hit = sub[sub["threshold_or_k"].astype(str) == thr_key]
            if not hit.empty:
                rec_row = hit.iloc[0]
        topk5_row: pd.Series | None = None
        s5 = sub.assign(_tk=pd.to_numeric(sub["topk_per_source"], errors="coerce"))
        s5 = s5[s5["_tk"] == 5]
        if not s5.empty:
            s5 = s5.copy()
            s5["_r"] = pd.to_numeric(s5["flow_pair_recall"], errors="coerce")
            s5["_f1"] = pd.to_numeric(s5["flow_pair_f1"], errors="coerce")
            topk5_row = s5.sort_values(["_r", "_f1"], ascending=[False, False]).iloc[0]
        _write_paper_results_decode_manuscript(exp, effective_rec=effective_rec, rec_row=rec_row, topk5_row=topk5_row)
    else:
        _write_paper_results_decode_manuscript(exp, effective_rec=effective_rec, rec_row=None, topk5_row=None)

    # --- 4) Paper-facing markdown package ---
    c12 = _sweep_row(out_root, "C_larger_budget")
    g18 = _sweep_row(out_root, "G_larger_budget_18M")
    f_oracle = _sweep_row(out_root, "F_oracle_upper_bound")

    syn_by = out_root / "eval" / "synthetic_eval_by_scenario.csv"
    syn_dbg = out_root / "eval" / "synthetic_failure_debug.csv"
    syn_txt = ""
    if syn_by.is_file():
        syn_txt = syn_by.read_text(encoding="utf-8")[:4000]

    pool_md = [
        "# Paper results: candidate pool budget (non-oracle vs oracle diagnostic)",
        "",
        "**Non-oracle rows** describe the **global_dst_pool** under fixed ``flow_max_matrix_cells`` / ``flow_dst_top_k``; "
        "metrics such as ``candidate_dst_recall`` are **retrieval** diagnostics, not RC-UOT solver accuracy on the full BNB universe.",
        "",
        "**Oracle diagnostic (``F_oracle_upper_bound``):** injects true destinations — **not** a fair baseline model; "
        "use only as an upper bound on recall given perfect retrieval.",
        "",
        f"- **12M (`C_larger_budget`):** candidate_dst_recall **{c12.get('candidate_dst_recall', 'n/a')}**, "
        f"bnb_active **{c12.get('bnb_active', 'n/a')}**, matrix_cells **{c12.get('matrix_cells', 'n/a')}**.",
        f"- **18M diagnostic (`G_larger_budget_18M`, see ``large_budget_setting.json``):** candidate_dst_recall **{g18.get('candidate_dst_recall', 'n/a')}**, "
        f"bnb_active **{g18.get('bnb_active', 'n/a')}**.",
        f"- **Oracle upper bound row:** candidate_dst_recall **{f_oracle.get('candidate_dst_recall', 'n/a')}** (diagnostic only).",
        "",
        "Source: ``experiments/candidate_pool_sweep.csv``.",
        "",
    ]
    (exp / "paper_results_candidate_pool.md").write_text("\n".join(pool_md), encoding="utf-8")

    def _scenario_cell(sdf: pd.DataFrame, scenario: str, col: str) -> str:
        sub = sdf[sdf.get("scenario", "").astype(str) == scenario]
        if sub.empty or col not in sub.columns:
            return "n/a"
        v = sub.iloc[0].get(col)
        s = str(v).strip()
        return s if s else "n/a"

    split_r = merge_r = smg = um_f1 = decoy = "n/a"
    if syn_by.is_file():
        try:
            sdf = pd.read_csv(syn_by, dtype=str, keep_default_na=False)
            split_r = _scenario_cell(sdf, "split", "edge_recovery_rate")
            merge_r = _scenario_cell(sdf, "merge", "edge_recovery_rate")
            smg = _scenario_cell(sdf, "split_merge_global", "edge_recovery_rate")
            um_f1 = _scenario_cell(sdf, "unmatched", "unmatched_detection_f1")
            decoy = _scenario_cell(sdf, "delay_noise", "decoy_pair_match_rate")
        except Exception:
            pass

    synth_md = [
        "# Paper results: semi-synthetic split / merge / unmatched / noise",
        "",
        "**Population (explicit):** rows below come from the **semi-synthetic** label/transport workspace "
        "(cloned segments and injected scenarios). They **do not** describe prevalence on the full real Celer export.",
        "",
        "## Structured recovery (split / merge / global)",
        "",
        f"- **split** ``edge_recovery_rate``: **{split_r}**",
        f"- **merge** ``edge_recovery_rate``: **{merge_r}**",
        f"- **split_merge_global** ``edge_recovery_rate``: **{smg}**",
        "",
        "When these values are near **1.0**, the transport plan recovers essentially all **synthetic template** edges that "
        "lie on the same small subgraph as the solve — a useful **sanity check** for the semi-synthetic generator and solver wiring, "
        "not a guarantee of analogous behavior on real one-to-one-dominated data.",
        "",
        "## Unmatched scenario (``unmatched_detection_f1``)",
        "",
        f"Table value: **{um_f1}**. Per ``experiments/synthetic_metric_interpretation.md``, this F1 compares "
        "hint-listed unmatched sources to ETH flows flagged by ``unmatched_ratio >= 0.08`` in ``uot_unmatched_mass.csv``. "
        "**A value of 0.0 means no true positives under that fixed gate**, commonly because USD-normalized unmatched ratios "
        "in the synthetic rows stay **far below 0.08** (see ``eval/synthetic_failure_debug.csv``), not because the solver "
        "assigns zero unmatched mass in absolute terms. This is primarily a **metric/eligibility alignment** issue for prose; "
        "**no solver code change is required** unless you intentionally redesign the unmatched detector or synthetic mass scale.",
        "",
        "## Delay-noise / decoys (``decoy_pair_match_rate``)",
        "",
        f"Table value: **{decoy}**. This is the **fraction of listed decoy edges that carry positive mass** in the exported "
        "transport plan — a **hit rate** on decoys. **1.0** means **every** decoy edge appears in the plan (poor decoy suppression "
        "in edge space). The name is **not inverted**: higher is worse for rejection. Derived ``synthetic_decoy_rejection_rate`` "
        "in bundled JSON uses **1 − hit rate**. Again, interpret as **evaluation signal**, not a second hidden bug, unless you "
        "choose to rename columns for readability.",
        "",
        "## Raw table excerpt",
        "",
        "```text",
        syn_txt if syn_txt else "_missing eval/synthetic_eval_by_scenario.csv_",
        "```",
        "",
        f"Row-level narrative: ``{syn_dbg.as_posix() if syn_dbg.is_file() else '(run --synthetic-failure-debug for eval/synthetic_failure_debug.csv)'}``.",
        "",
    ]
    (exp / "paper_results_synthetic_failure.md").write_text("\n".join(synth_md), encoding="utf-8")

    lim_md = [
        "# Paper limitations (refined)",
        "",
        "## Real Celer weak labels",
        "",
        "Public bridge-facing flows are **predominantly one-to-one** at the segment layer. Claims about **split, merge, "
        "and unmatched** handling on real Celer data should be **limited** to coverage/candidate-pool and decoding metrics, "
        "unless rare multi-edge labels are explicitly quantified.",
        "",
        "## Semi-synthetic evidence",
        "",
        "Complex structural behaviors (split/merge/unmatched/delay-noise decoys) are evaluated mainly on **semi-synthetic** "
        "subgraphs (``eval/synthetic_eval_by_scenario.csv`` and related UOT workspace). These results **do not** substitute "
        "for prevalence claims on the full real dataset.",
        "",
        "## Oracle diagnostics",
        "",
        "``F_oracle_upper_bound`` and related oracle tooling establish **feasibility / recall ceilings** when true destinations "
        "are forced into the pool. They are **not** competing matchers in the main comparison table.",
        "",
        "## Non-oracle bottleneck",
        "",
        "``candidate_dst_recall`` under fixed matrix budgets shows that **candidate retrieval** can dominate end-to-end "
        "weak-label performance. RC-UOT vs Hungarian/Greedy should be compared **conditional on the same active BNB pool**.",
        "",
    ]
    (exp / "paper_limitations_refined.md").write_text("\n".join(lim_md), encoding="utf-8")

    logger.info(
        "paper_finalization: audit=%s; wrote final_artifact_*, decode_threshold_*, paper_results_*, "
        "paper_limitations_refined.md, synthetic_metric_interpretation.md",
        "PASS" if all_passed else "FAIL",
    )
    return 0 if all_passed else 1
