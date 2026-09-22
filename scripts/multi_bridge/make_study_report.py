"""Assemble the FINAL_BASELINE_MECHANISM_REPORT.md + RUN_MANIFEST.md for the study."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

STUDY = REPO / "out" / "multi_bridge_expansion" / "structural_baseline_mechanism_study"


def fmt(v: Any, nd: int = 3) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def main() -> int:
    main_df = pd.read_csv(STUDY / "aggregated" / "main_structural_comparison.csv")
    stress = pd.read_csv(STUDY / "stress" / "stress_aggregated.csv")
    sel = json.loads((STUDY / "calibration" / "selected_threshold.json").read_text(encoding="utf-8"))
    verify = json.loads((STUDY / "verification" / "verification_report.json").read_text(encoding="utf-8"))
    cap = pd.read_csv(STUDY / "capability_tests" / "capability_matrix.csv")

    def row(br: str, m: str) -> dict[str, Any]:
        r = main_df[(main_df["bridge"] == br) & (main_df["method"] == m)]
        return r.iloc[0].to_dict() if not r.empty else {}

    lines: list[str] = []
    A = lines.append

    A("# FINAL BASELINE MECHANISM REPORT")
    A("")
    A("Study directory: `out/multi_bridge_expansion/structural_baseline_mechanism_study/` "
      "(the frozen faithful pipeline directory was not modified).")
    A("Date: 2026-09-01 (Asia/Shanghai). RC-UOT-Q candidate main result: FROZEN — the frozen "
      "transport plans were decoded, never re-solved, for the main comparison; stress ladders "
      "run RC-UOT-Q on new data with the identical frozen parameters.")
    A("")

    A("## 1. Acceptance answers (the 20 questions)")
    A("")
    A("**1. Connector-style strictly one-to-one at code level?** Source-side yes, target-side "
      "no. `WithdrawLocator.search_withdraw` forces source out-degree ≤ 1 "
      "(`scoped.groupby(txhash)['time_diff'].idxmin()` + `drop_duplicates(subset=['txhash'])`); "
      "no destination-uniqueness constraint exists. Verified by unit tests + verification audit "
      "(max out-degree 1 across all seeds).")
    A("**2. ABCTracer-style strictly one-to-one?** Same answer: per-source argmax in "
      "`predict_abctracer_style` → source out-degree ≤ 1; no target-degree constraint.")
    A("**3. What does 'Unmatched = PARTIAL' mean for them?** Rejection-level abstention only: "
      "Connector emits an empty destination only when every filter stage (receiver/tx-type/"
      "asset/timestamp/amount) rejects all candidates; ABCTracer returns None only when the time "
      "window is empty or the best score ≤ 0. Neither has first-class abstention, and neither "
      "can represent an unmatched destination. Code evidence in "
      "`capability_tests/unmatched_semantics.md`. The original table cell is ACCURATE.")
    A("**4. Did Threshold-MM actually emit 1→2 and 2→1?** Yes. Verification audit "
      "`decoded_has_1to2`/`decoded_has_2to1` = true for Threshold-MM, Balanced-OT, RC-UOT-Q on "
      "all bridge/seed grids; false for Connector/ABCTracer on 1→2 (their 2→1 is true because "
      "their target degree is unconstrained).")
    A(f"**5. Global tau?** τ* = {sel['global_tau']} (cost cutoff {sel['global_cutoff_cost']:.6f}), "
      "selected by pooled edge F1 on calibration seeds 101–103 (432 templates, 746,496 cost "
      "cells, pre-registered grid {0.05…0.95} of the calibration cost ECDF). Test seeds 42–46 "
      "were never read by calibration (asserted in `selected_threshold.json`). Note: τ* sits at "
      "the grid's lower boundary (edge F1 is monotone decreasing in τ); no below-grid search was "
      "performed. Per-bridge τ* are reported supplementary; the main baseline uses the global τ.")
    A("**6. Did Balanced-OT actually emit 1→2 / 2→1?** Yes — the entropic balanced plan decoded "
      "by the same mass threshold (1e-9, never Hungarian) emits multi-destination sources and "
      "multi-source targets on every grid; also verified by the minimal unit tests TEST B/C.")
    A("**7. Is Balanced-OT strictly global allocation?** Yes — `min_P <C,P> + reg·H(P)` s.t. "
      "P1 = a, Pᵀ1 = b over the joint plan (TEST F: S1→D1 mass moves 0.25 → 0.0013 when "
      "competing mass changes, while Threshold-MM's local decision is invariant).")
    A("**8. Can Balanced-OT abstain without a dustbin?** No. With strictly balanced marginals "
      "every unit of mass is transported: TEST D shows the unmatched source's mass flows out "
      "fully and TEST E shows the unmatched target receives forced mass. Decoded plans are "
      "dense (mean ~36% of cells ≥ 1e-9 on the main grids).")
    A("**9. Does RC-UOT-Q show unmatched / residual mass?** Yes — mass-level. Frozen Celer "
      "seed-42: transported 0.782 of 1.0 (0.218 residual). Caveat verified in TEST D/E: at the "
      "literal 1e-9 edge threshold the decoded edge set can still contain a low-mass edge for "
      "the unmatched node; abstention is a mass-level mechanism, threshold-dependent at edge level.")
    A("**10. Three-bridge × five-method mean ± std.** See Table 2d in the manuscript and "
      "`aggregated/main_structural_comparison.md`. Split/merge exact: 0.000 ± 0.000 for every "
      "method and bridge. Edge F1: Celer Threshold-MM 0.168 ± 0.016 > RC-UOT-Q 0.040 ± 0.009 > "
      "Balanced-OT 0.030 ± 0.005 > Connector/ABC ≈ 0.026; Multichain 0.136 / 0.027 / 0.018 / "
      "≈0.017–0.034; Poly 0.120 / 0.034 / 0.026 / 0.000. FP edges per template: threshold 67.7–"
      "107.5; RC-UOT-Q 539–925; Balanced-OT 681–1070; one-to-one ≈ 5.8–6.0. Degree accuracy ≈ 0 "
      "for all.")
    A("**11. Clean conditions: Balanced-OT vs RC-UOT-Q.** Not close (pre-registered H3 "
      "unsupported): edge F1 at 0% mismatch — Celer 0.070 vs 0.110; Multichain 0.032 vs 0.047; "
      "Poly 0.033 vs 0.045. RC-UOT-Q already leads at the balanced operating point.")
    A("**12. Mass mismatch curves.** Both transport methods degrade (Celer precision: "
      "Balanced-OT 0.043→0.024; RC-UOT-Q 0.068→0.037 over m = 0→0.40), Threshold-MM is flat/"
      "improving (0.142→0.155; the inflated destination amounts amplify its pairwise amount "
      "signal). RC-UOT-Q stays above Balanced-OT at every level; the advantage is a level "
      "offset, not a slope difference.")
    A("**13. Unmatched ratio curves.** All three methods lose precision (Celer at 40%: "
      "Threshold-MM 0.086, Balanced-OT 0.029, RC-UOT-Q 0.035); RC-UOT-Q retains its offset over "
      "Balanced-OT at every level while destroying the excess mass.")
    A("**14. Decoy growth of Threshold-MM false positives.** Yes, clearly: Celer FP/template "
      "26.3 (k=1) → 223.6 (k=8), +750%, precision 0.182 → 0.061. Joint methods grow more slowly "
      "(Balanced-OT +278%, RC-UOT-Q +239%) but keep higher absolute FP because their decodes "
      "are dense (H2's relative claim supported).")
    A("**15. Any result violating the original capability table?** Yes, two corrections "
      "(documented in `capability_tests/capability_table.md`): (a) 'Split/Merge capable = NO' "
      "for Connector/ABCTracer is imprecise for merge — their code can represent 2→1 (no "
      "target-degree constraint); corrected to NO (split) / PARTIAL (merge). (b) The original "
      "manuscript claim 'a one-to-one matcher cannot emit … a 2→1 merge' is false in general; "
      "their zero merge recovery here is an empirical outcome of pairwise amount/time selection. "
      "All other cells are confirmed.")
    A("**16. Final capability table.** As Table 2c in the manuscript:")
    A("")
    A("| Method | Split | Merge | Global allocation | Unmatched |")
    A("| --- | --- | --- | --- | --- |")
    A("| Connector-style | NO | PARTIAL | NO | PARTIAL |")
    A("| ABCTracer-style | NO | PARTIAL | NO | PARTIAL |")
    A("| Threshold-MM | YES | YES | NO | YES |")
    A("| Balanced-OT | YES | YES | YES | NO |")
    A("| RC-UOT-Q | YES | YES | YES | YES |")
    A("")
    A("**17. Highest supported claim level: LEVEL 1** — RC-UOT-Q removes the representational "
      "limitation of one-to-one hard matching (unit-test-verified). LEVEL 2 is NOT supported: "
      "the calibrated Threshold-MM dominates edge F1 (0.120–0.168 vs 0.027–0.040), so 'global "
      "transport improves structural consistency over independent pairwise expansion' is not "
      "established on this benchmark. LEVEL 3 is NOT supported: Balanced-OT is not close to "
      "RC-UOT-Q even clean, exact recovery is 0 for both everywhere, and the unbalanced-"
      "robustness evidence is a consistent edge-metric offset rather than exact-topology "
      "robustness. The honest subsidiary finding: RC-UOT-Q > strictly balanced OT at every "
      "bridge and every stress level on edge precision/F1.")
    A("**18. Final figure paths.** "
      f"`{STUDY}/figures/figureA_capability_ladder.png` (also .pdf/.svg), "
      f"`{STUDY}/figures/figureB_three_bridge_structural.png`, "
      f"`{STUDY}/figures/figureC_mechanism_stress.png`, "
      f"`{STUDY}/figures/figureS1_fp_decoy_ladder.png`; manuscript copies at "
      "`manuscript_final/figures/ch4/fig5{c,d,e,f}_*.png`.")
    A("**19. Reproduction commands.** See `RUN_MANIFEST.md` (same directory).")
    A("**20. Table corrections applied.** Yes — the manuscript's capability statements and "
      "Table 2b/4.3 have been revised to the corrected table (merge = PARTIAL) and to the "
      "honest strict-evaluation numbers; the frozen RC-UOT-Q edge-inclusion numbers are kept "
      "and explicitly relabeled as such.")

    A("")
    A("## 2. Headline numbers")
    A("")
    A("| Bridge | Method | Split exact | Merge exact | Edge F1 | Edge P | Edge R | FP/tpl |")
    A("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for br in ("Celer", "Multi", "Poly"):
        for m in ("Connector-style", "ABCTracer-style", "Threshold-MM", "Balanced-OT", "RC-UOT-Q"):
            r = row(br, m)
            if not r:
                continue
            A(f"| {br} | {m} | {fmt(r['split_exact_mean'])} | {fmt(r['merge_exact_mean'])} | "
              f"{fmt(r['edge_f1_mean'])} ± {fmt(r['edge_f1_std'])} | "
              f"{fmt(r['edge_precision_mean'])} | {fmt(r['edge_recall_mean'])} | "
              f"{fmt(r['fp_edges_mean'], 1)} |")
    A("")
    A(f"Calibration: global τ* = {sel['global_tau']} (cutoff {sel['global_cutoff_cost']:.6f}); "
      "calibration F1 at τ* = "
      f"{sel['selected_grid_row']['edge_f1']:.4f} (P {sel['selected_grid_row']['edge_precision']:.4f}, "
      f"R {sel['selected_grid_row']['edge_recall']:.4f}).")
    A("")
    A(f"Independent verification: **{verify['overall_pass']}** "
      f"({verify['n_issues']} issues; seeds 5/5 per bridge; 48 templates per seed; no NaN; "
      "cross-method template identity holds; recomputed metrics match the aggregated tables).")
    A("")
    A("## 3. What the experiments actually established")
    A("")
    A("1. **Representation (Layer 1):** capability differences are real and code-verified — "
      "the one-to-one decoders cannot emit 1→2 (structural), can represent 2→1 but score 0 on "
      "this benchmark, and abstain only by rejection; the many-match methods emit both "
      "structures, and the OT methods allocate globally (TEST F) with Balanced-OT unable to "
      "abstain (TEST D/E).")
    A("2. **Strict evaluation (Layer 2):** at the frozen operating point (decode threshold "
      "1e-9) every method's exact topology recovery is 0.000 on every bridge. Causes: dense "
      "decodes; truth edges at per-source cost rank ≈3–4 (confusers cheaper per-edge); legacy "
      "metric = edge inclusion. Edge-level ranking: Threshold-MM > RC-UOT-Q > Balanced-OT > "
      "one-to-one baselines.")
    A("3. **Mechanism (Layer 3):** mass mismatch and unmatched ratios degrade both transport "
      "methods (RC-UOT-Q keeps a level offset above Balanced-OT); decoy density raises "
      "Threshold-MM's FPs fastest (pairwise independence, H2); timestamp noise at the tested "
      "scales does not discriminate any method.")
    A("4. **Conclusion level: LEVEL 1**, with the subsidiary mechanism statement that "
      "RC-UOT-Q consistently outperforms strictly balanced OT on edge metrics under every "
      "stress level. The manuscript §4.3 and §4.8 have been rewritten accordingly "
      "(nature-writing/polishing skills were unavailable in this session; their standards — "
      "hedged claims, explicit evidence pointers, no overclaiming — were applied manually).")

    (STUDY / "FINAL_BASELINE_MECHANISM_REPORT.md").write_text("\n".join(lines) + "\n",
                                                              encoding="utf-8")
    print("report written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
