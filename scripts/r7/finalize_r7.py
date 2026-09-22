"""R7 finalisation driver: analysis -> decision -> paper -> figures -> checklist -> report
-> manifest.

Read-only with respect to the confirmatory raw package (re-reads it; never regenerates).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r7.r7_analysis import analyse
from r7.r7_common import DIR_CONFIRMATORY, DIR_FIGURES, DIR_RAW, DIR_SELECTION, log
from r7.r7_figures import build_all as build_figures
from r7.r7_reporting import (build_checklist, build_decision, build_final_report,
                             build_manifest, build_paper, build_selection_data_manifest)


def main() -> int:
    gate_e = DIR_CONFIRMATORY / "VALIDITY_GATE_E.json"
    log("[final] running confirmatory analysis")
    analysis = analyse(DIR_RAW, gate_e)

    log("[final] DECISION.json")
    decision = build_decision(analysis)
    log(f"[final] classification = {decision['classification']}")

    log("[final] selection + confirmatory data manifest")
    dm = build_selection_data_manifest()
    log(f"[final] data_manifest: selection {dm['selection']['n_cells']} cells, "
        f"confirmatory {dm['confirmatory']['n_cells']} cells")

    log("[final] manuscript patches")
    build_paper(analysis, decision)

    log("[final] figures")
    figures = build_figures(analysis)

    log("[final] manifest (pre-report pass)")
    manifest = build_manifest()

    log("[final] validation checklist")
    build_checklist(analysis, decision, figures)

    log("[final] final experiment report")
    build_final_report(analysis, decision, figures, manifest)

    log("[final] manifest (final pass, includes the report)")
    manifest = build_manifest()

    print(json.dumps({
        "classification": decision["classification"],
        "manuscript_branch": decision["manuscript_branch"],
        "gate_A": decision["gate_A"]["PASS"], "gate_B": decision["gate_B"]["PASS"],
        "gate_C": decision["gate_C"]["PASS"], "gate_D": decision["gate_D"]["PASS"],
        "gate_E": decision["gate_E"]["PASS"],
        "manifest_files": manifest["n_files"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
