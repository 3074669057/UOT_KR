"""One-factor DIAGNOSTIC sweeps on development seeds 201-205 ONLY.

- reg sweep {0.005, 0.01, 0.02, 0.05, 0.10, 0.20} (0.05 = frozen reference), UOT and BOT,
  all else frozen.
- reg_m sweep {0.10, 0.25, 0.50, 1.00, 2.00, 5.00} (0.50 = frozen), UOT only, reg=0.05.
- marginal counterfactuals {frozen, amount-only, uniform} (DIAGNOSTIC), UOT and BOT.

Every output is DIAGNOSTIC ONLY — never method selection, never tuned on 42-46, never
run on 301-305. Usage: python run_dev_sweeps.py --bridge X [--what reg,regm,marginal]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from diag.ctd_common import (  # noqa: E402
    BRIDGES, CTD, DEV_SEEDS, FROZEN_REG, FROZEN_REGM, REGM_GRID, REG_GRID,
    load_dev_cell, marginal_variants, solve_bot_log, solve_uot_log,
)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default=None)
    ap.add_argument("--what", default="reg,regm,marginal")
    cli = ap.parse_args()
    bridges = (cli.bridge,) if cli.bridge else BRIDGES
    what = set(cli.what.split(","))

    for bridge in bridges:
        for seed in DEV_SEEDS:
            cell = load_dev_cell(bridge, seed)
            C = cell["C"]
            if "reg" in what:
                for reg in REG_GRID:
                    out = CTD / "reg_sweep" / bridge / f"seed_{seed}"
                    out.mkdir(parents=True, exist_ok=True)
                    tag = f"reg_{str(reg).replace('.', 'p')}"
                    cache = out / f"{tag}_meta.json"
                    if cache.is_file():
                        continue
                    u = solve_uot_log(cell["a_rw"], cell["b_ev"], C, reg, FROZEN_REGM)
                    b = solve_bot_log(cell["a_rw"], cell["b_ev"], C, reg)
                    np.savez(out / f"uot_{tag}.npz", P=u["P"], logu_raw=u["logu_raw"],
                             logv_raw=u["logv_raw"])
                    np.savez(out / f"bot_{tag}.npz", P=b["P"], u_pot=b["u_pot"], v_pot=b["v_pot"])
                    cache.write_text(json.dumps({
                        "reg": reg,
                        "uot": {k: float(v) for k, v in u.items() if k == "converged" or k == "final_err"},
                        "bot": {k: float(v) for k, v in b.items()
                                if k in ("converged", "final_err", "row_residual", "col_residual")},
                        "diagnostic_only": True}, indent=1, default=str) + "\n", encoding="utf-8")
                    print(f"[reg] {bridge} s{seed} reg={reg}: uot_conv={u['converged']} "
                          f"bot_conv={b['converged']}", flush=True)
            if "regm" in what:
                for reg_m in REGM_GRID:
                    out = CTD / "reg_m_sweep" / bridge / f"seed_{seed}"
                    out.mkdir(parents=True, exist_ok=True)
                    tag = f"regm_{str(reg_m).replace('.', 'p')}"
                    cache = out / f"{tag}_meta.json"
                    if cache.is_file():
                        continue
                    u = solve_uot_log(cell["a_rw"], cell["b_ev"], C, FROZEN_REG, reg_m)
                    np.savez(out / f"uot_{tag}.npz", P=u["P"], logu_raw=u["logu_raw"],
                             logv_raw=u["logv_raw"])
                    cache.write_text(json.dumps({
                        "reg_m": reg_m,
                        "uot": {k: float(v) for k, v in u.items() if k == "converged" or k == "final_err"},
                        "diagnostic_only": True}, indent=1, default=str) + "\n", encoding="utf-8")
                    print(f"[regm] {bridge} s{seed} reg_m={reg_m}: conv={u['converged']} "
                          f"residual_mass={1.0 - float(u['P'].sum()):.4f}", flush=True)
            if "marginal" in what:
                variants = marginal_variants(cell)
                for vname, (a, b) in variants.items():
                    out = CTD / "marginals" / vname / bridge
                    out.mkdir(parents=True, exist_ok=True)
                    cache = out / f"seed_{seed}_meta.json"
                    if cache.is_file():
                        continue
                    u = solve_uot_log(a, b, C, FROZEN_REG, FROZEN_REGM)
                    bb = solve_bot_log(a, b, C, FROZEN_REG)
                    np.savez(out / f"seed_{seed}_uot.npz", P=u["P"], logu_raw=u["logu_raw"],
                             logv_raw=u["logv_raw"], a=a, b=b)
                    np.savez(out / f"seed_{seed}_bot.npz", P=bb["P"], u_pot=bb["u_pot"],
                             v_pot=bb["v_pot"], a=a, b=b)
                    cache.write_text(json.dumps({
                        "variant": vname,
                        "uot": {k: float(v) for k, v in u.items() if k == "converged" or k == "final_err"},
                        "bot": {k: float(v) for k, v in bb.items()
                                if k in ("converged", "final_err", "row_residual", "col_residual")},
                        "diagnostic_only": True}, indent=1, default=str) + "\n", encoding="utf-8")
                    print(f"[marginal] {bridge} s{seed} {vname}: uot_conv={u['converged']} "
                          f"bot_conv={bb['converged']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
