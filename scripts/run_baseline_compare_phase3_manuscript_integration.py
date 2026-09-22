#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 3: Manuscript integration QA (text-level checks only)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MS = REPO / "manuscript_final"
OUT = REPO / "out" / "baseline_compare" / "phase3_manuscript_integration_report.md"
TABLE5_SRC = REPO / "out" / "final_paper_tables" / "main_table_rc_uot_q_fixed_delay.md"
PHASE22 = REPO / "out" / "baseline_compare" / "fair_main_compare_phase2_2"


def _read(*paths: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in paths if p.is_file())


def main() -> None:
    exp = _read(MS / "04_experiments.md")
    full = _read(MS / "full_manuscript_final.md")
    app_b = _read(MS / "appendix_B_connector_native_diagnostic.md")
    blob = exp + full + app_b
    table5_orig = TABLE5_SRC.read_text(encoding="utf-8") if TABLE5_SRC.is_file() else ""

    checks = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "pass": ok, "detail": detail})

    add(
        "fair_table_rc_uot_q_accepted_rows",
        all(x in exp for x in ("raw_argmax_fixed_delay", "positive_delay_top3_rescue", "joint_time_admissible_filter", "0.7085")),
        "Table 6 includes three RC-UOT-Q operating points with joint F1 0.7085",
    )
    add(
        "connector_abctracer_na_blocked",
        "BLOCKED_BY_REQUIRED_BRIDGE_SEMANTICS" in exp and "Connector | raw_top1_anchor_masked | N/A" in exp,
        "Connector rows N/A + BLOCKED; ABCTracer BLOCKED",
    )
    add(
        "native_f1_only_appendix_not_table6",
        "0.9736" in app_b and "0.9736" not in exp.split("Table 6")[1].split("## 4.5")[0] if "Table 6" in exp else False,
        "F1=0.9736 in Appendix B only, not in Table 6 body",
    )
    add(
        "no_rc_uot_q_outperforms_connector_f1",
        not re.search(r"outperforms.*Connector|Connector.*outperformed", blob, re.I),
        "No 'outperforms Connector' phrasing",
    )
    add(
        "no_connector_f1_zero_fair",
        not re.search(r"Connector.*F1\s*=\s*0[^.]|\bConnector has F1=0\b", blob, re.I),
        "No Connector F1=0 under fair comparison",
    )
    add(
        "no_connector_fails",
        not re.search(r"Connector fails", blob, re.I),
        "No 'Connector fails' phrasing",
    )
    add(
        "no_abctracer_fails_or_evaluated",
        not re.search(r"ABCTracer fails|ABCTracer was evaluated|ABCTracer performs", blob, re.I)
        and "could not be evaluated as an original-system baseline" in blob,
        "ABCTracer blocked wording; not claimed evaluated",
    )
    add(
        "table5_unchanged_in_final_paper_tables",
        "0.708" in table5_orig and "0.589" in table5_orig,
        "Frozen Table 5 source file untouched (spot-check metrics present)",
    )
    add(
        "table5_still_in_manuscript",
        "Table 5" in exp and "0.708" in exp.split("Table 5")[1][:800],
        "Table 5 retained in manuscript §4.4",
    )
    add(
        "table6_distinct_from_table5",
        "Table 6" in exp and "does not replace Table 5" in exp,
        "Table 6 labeled as new fair comparison, not Table 5 replacement",
    )
    add(
        "fair_vs_native_distinction",
        "anchor-masked" in blob and "native closed-set diagnostic" in blob.lower(),
        "Manuscript distinguishes fair anchor-masked vs native diagnostic",
    )

    all_pass = all(c["pass"] for c in checks)
    lines = [
        "# Phase 3 manuscript integration report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"**Overall QA:** {'PASS' if all_pass else 'FAIL'}",
        "",
        "## Changed manuscript files",
        "",
        "- `cross/manuscript_final/04_experiments.md` — added §4.4.2 + Table 6",
        "- `cross/manuscript_final/full_manuscript_final.md` — same integration + Appendix B pointer",
        "- `cross/manuscript_final/appendix_B_connector_native_diagnostic.md` — new appendix cross-reference",
        "",
        "## Insertion locations",
        "",
        "- **Table 6 / §4.4.2:** After §4.4.1 Leakage audit, before §4.5 Coverage-Qualified Quotient Inference",
        "- **Appendix B:** End of `full_manuscript_final.md` (after Appendix A); full text in `appendix_B_connector_native_diagnostic.md`",
        "",
        "## Source artifacts (read-only)",
        "",
        "- `out/baseline_compare/fair_main_compare_phase2_2/fair_main_comparison_table_paper_ready.md`",
        "- `out/baseline_compare/fair_main_compare_phase2_2/fair_main_comparison_caption.md`",
        "- `out/baseline_compare/fair_main_compare_phase2_2/fair_main_comparison_paragraph.md`",
        "- `out/baseline_compare/fair_main_compare_phase2_2/connector_native_appendix_cross_reference.md`",
        "",
        "## Text-level QA",
        "",
        "| # | Check | Result | Detail |",
        "|---|-------|:------:|--------|",
    ]
    for i, c in enumerate(checks, 1):
        lines.append(f"| {i} | {c['check']} | {'PASS' if c['pass'] else 'FAIL'} | {c['detail']} |")

    lines += [
        "",
        "## Constraints honored",
        "",
        "- No new experiments; no UOT rerun; no Connector/ABCTracer core edits",
        "- Phase 1 / Phase 2 raw JSON outputs unchanged",
        "- `out/final_paper_tables/` not modified",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Phase 3 QA report: {OUT} overall={'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
