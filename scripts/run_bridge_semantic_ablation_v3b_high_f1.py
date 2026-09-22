#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bridge semantic ablation v3b — supplementary high-F1 operating-point search.

Outputs ONLY to: out/baseline_compare/bridge_semantic_ablation_v3b_high_f1/
Does NOT modify v3 main results or production base.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cross.baseline_compare.v3b_high_f1.acceptance import (  # noqa: E402
    run_acceptance,
    save_integrity_baselines,
)
from cross.baseline_compare.v3b_high_f1.constants import (  # noqa: E402
    BNB_CSV,
    ETH_CSV,
    LABELS,
    OUT_V3,
    OUT_V3B,
    UOT_BASE,
)
from cross.baseline_compare.v3b_high_f1.eval_test import eval_test  # noqa: E402
from cross.baseline_compare.v3b_high_f1.split import make_split  # noqa: E402
from cross.baseline_compare.v3b_high_f1.subset_analysis import run_subset_analysis  # noqa: E402
from cross.baseline_compare.v3b_high_f1.summary import summarize  # noqa: E402
from cross.baseline_compare.v3b_high_f1.tune_dev import tune_dev  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Bridge semantic ablation v3b high-F1 supplementary.")
    ap.add_argument("--make-split", action="store_true")
    ap.add_argument("--tune-dev", action="store_true")
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--finish", action="store_true", help="eval-test + subset + summarize (resume-friendly).")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_V3B)
    args = ap.parse_args()

    if not any([args.make_split, args.tune_dev, args.eval_test, args.summarize, args.finish, args.all]):
        ap.error("Specify --make-split, --tune-dev, --eval-test, --summarize, --finish, or --all.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_integrity_baselines(out, OUT_V3, UOT_BASE)

    if args.make_split or args.all:
        manifest = make_split(labels_dir=LABELS, output_dir=out)
        print(f"Split: dev={manifest['n_dev_pairs']} test={manifest['n_test_pairs']}")

    if args.tune_dev or args.all:
        tune_dev(output_dir=out, eth_csv=ETH_CSV, bnb_csv=BNB_CSV)

    if args.eval_test or args.all or args.finish:
        eval_test(output_dir=out, eth_csv=ETH_CSV, bnb_csv=BNB_CSV)

    if args.all or args.finish:
        run_subset_analysis(output_dir=out, eth_csv=ETH_CSV, bnb_csv=BNB_CSV)

    if args.summarize or args.all or args.finish:
        summarize(output_dir=out, v3_dir=OUT_V3)

    acc = run_acceptance(output_dir=out, v3_dir=OUT_V3, uot_base=UOT_BASE)
    print(f"v3b acceptance: {'PASS' if acc['all_pass'] else 'FAIL'}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
