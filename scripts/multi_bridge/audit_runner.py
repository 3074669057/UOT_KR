"""Audit runner: re-run the frozen synthetic inputs through the SAME pipeline with
pre-registered variants (ablations, sensitivity, perturbations, counterfactuals).

All inputs are COPIED from the candidate main run (never regenerated), so the
synthetic subgraphs are bit-identical to the main run for every (bridge, seed).
Outputs go ONLY under out/multi_bridge_expansion/faithful_flow_structural_three_bridges_audit/.
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cross.application.standalone_flow_uot import run_standalone_flow_uot  # noqa: E402
from cross.domain.uot.cost_matrix import default_cost_weights  # noqa: E402

MAIN = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges"
AUDIT = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges_audit"
INPUTS = AUDIT / "synthetic_inputs"
BRIDGES = ("Celer", "Multi", "Poly")
SEEDS = (42, 43, 44, 45, 46)

BASE_WEIGHTS = default_cost_weights()  # amount .35 time .25 route .15 risk .15 graph .05 evidence .05 novelty .05


def renormalize_weights(w: dict[str, float]) -> dict[str, float]:
    tot = sum(float(v) for v in w.values())
    if tot <= 0:
        return {k: 0.0 for k in w}
    return {k: float(v) / tot for k, v in w.items()}


def weights_loo(comp: str) -> dict[str, float]:
    """all - <comp>, renormalized (paper's ablation semantics)."""
    w = {k: float(v) for k, v in BASE_WEIGHTS.items()}
    if comp == "graph":
        w["graph"] = 0.0
    else:
        w[comp] = 0.0
    return renormalize_weights(w)


def weights_only(comp: str) -> dict[str, float]:
    w = {k: 0.0 for k in BASE_WEIGHTS}
    w[comp] = 1.0
    return w


def weights_minimum_defensible() -> dict[str, float]:
    return renormalize_weights({"amount": 0.35, "time": 0.25, "route": 0.15})


def ensure_inputs() -> None:
    for br in BRIDGES:
        for seed in SEEDS:
            src_dir = MAIN / "per_seed" / br / f"seed_{seed}"
            dst_dir = INPUTS / br / f"seed_{seed}"
            for rel in ("flow_segments_eth_synth.csv", "flow_segments_bnb_synth.csv"):
                p = src_dir / rel
                if not p.is_file():
                    raise FileNotFoundError(f"missing main-run input: {p}")
                dst = dst_dir / rel
                if not dst.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dst)
            lab_dir = dst_dir / "labels"
            lab_dir.mkdir(parents=True, exist_ok=True)
            for rel in ("synthetic_flow_labels.csv", "synthetic_uot_eval_metrics.json"):
                dst = lab_dir / rel
                if not dst.is_file():
                    shutil.copy2(src_dir / "labels" / rel, dst)


# ---------------------------------------------------------------- variant transforms
def _load(br: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    d = INPUTS / br / f"seed_{seed}"
    eth = pd.read_csv(d / "flow_segments_eth_synth.csv", dtype=str, keep_default_na=False)
    bnb = pd.read_csv(d / "flow_segments_bnb_synth.csv", dtype=str, keep_default_na=False)
    lab = pd.read_csv(d / "labels" / "synthetic_flow_labels.csv", dtype=str, keep_default_na=False)
    hints = json.loads((d / "labels" / "synthetic_uot_eval_metrics.json").read_text(encoding="utf-8"))
    return eth, bnb, lab, hints


def _scale_usd(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    out = df.copy()
    out["usd_amount_sum"] = (pd.to_numeric(out["usd_amount_sum"], errors="coerce").fillna(0.0) * factor).map(lambda x: f"{x:.12g}")
    return out


def apply_variant(br: str, seed: int, variant: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """variant keys: price_uniform, price_per_token_seed, ts_noise_std, amt_noise_level,
    decoy_mult, unmatched_mult, aml_zero, evidence_const, decoy_off."""
    eth, bnb, lab, hints = _load(br, seed)
    rng = random.Random(10_000 + seed)

    if "price_uniform" in variant:
        f = float(variant["price_uniform"])
        eth, bnb = _scale_usd(eth, f), _scale_usd(bnb, f)

    if "price_per_token_seed" in variant:
        rng_p = random.Random(20_000 + int(variant["price_per_token_seed"]))
        for df in (eth, bnb):
            toks = sorted(set(str(t) for t in df["token_contracts"] if str(t).strip()))
            factors = {t: float(2 ** rng_p.uniform(-0.322, 0.322)) for t in toks}  # [0.8, 1.25]
            df = _apply_token_factors(df, factors)
            if df is eth:
                eth = df
            else:
                bnb = df

    if "ts_noise_std" in variant:
        sig = float(variant["ts_noise_std"])
        rng_t = random.Random(30_000 + seed)
        for df in (eth, bnb):
            df = _jitter_times(df, sig, rng_t)
            if df is eth:
                eth = df
            else:
                bnb = df

    if "amt_noise_level" in variant:
        lv = float(variant["amt_noise_level"])
        rng_a = random.Random(40_000 + seed)
        for df in (eth, bnb):
            df = _amount_noise(df, lv, rng_a)
            if df is eth:
                eth = df
            else:
                bnb = df

    if "decoy_mult" in variant:
        k = int(variant["decoy_mult"])
        eth, bnb, lab, hints = _add_decoys(eth, bnb, lab, hints, mult=k, rng=rng)

    if "unmatched_mult" in variant:
        k = int(variant["unmatched_mult"])
        eth, bnb, lab, hints = _add_unmatched(eth, bnb, lab, hints, mult=k, rng=rng)

    if variant.get("aml_zero"):
        eth["aml_score_mean"] = "0"
        eth["aml_score_max"] = "0"

    if variant.get("evidence_const"):
        eth["evidence_quality_mean"] = "0.825"
        bnb["evidence_quality_mean"] = "0.675"

    if variant.get("decoy_off"):
        # counterfactual: remove the +60/+120 decoy offsets (timestamps back to template)
        bnb.loc[bnb["flow_id"].astype(str).str.contains("__synth_noise_dst"), ["start_time", "end_time"]] = bnb[
            bnb["flow_id"].astype(str).str.contains("__synth_noise_dst")
        ][["start_time", "end_time"]].apply(lambda s: pd.to_numeric(s, errors="coerce") - pd.to_numeric(s, errors="coerce").map(
            lambda _x: 0.0))
        # simpler: recompute from labels' median_delay? no — subtract the known offsets 60/120:
        idx = bnb["flow_id"].astype(str).str.contains("__synth_noise_dst")
        off = bnb.loc[idx, "flow_id"].astype(str).str.extract(r"noise_dst_(\d)")[0].map(lambda d: 60 + 60 * int(d))
        bnb.loc[idx, "start_time"] = (pd.to_numeric(bnb.loc[idx, "start_time"], errors="coerce") - off).map(lambda x: str(int(round(x))))
        bnb.loc[idx, "end_time"] = (pd.to_numeric(bnb.loc[idx, "end_time"], errors="coerce") - off).map(lambda x: str(int(round(x))))

    return eth, bnb, lab, hints


def _apply_token_factors(df: pd.DataFrame, factors: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    usd = pd.to_numeric(out["usd_amount_sum"], errors="coerce").fillna(0.0)
    for tok, f in factors.items():
        m = out["token_contracts"].astype(str) == tok
        usd[m] = usd[m] * f
    out["usd_amount_sum"] = usd.map(lambda x: f"{x:.12g}")
    return out


def _jitter_times(df: pd.DataFrame, sigma: float, rng: random.Random) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    jit = [rng.gauss(0.0, sigma) for _ in range(n)]
    for col in ("start_time", "end_time"):
        out[col] = (pd.to_numeric(out[col], errors="coerce").fillna(0.0) + jit).map(lambda x: str(int(round(x))))
    return out


def _amount_noise(df: pd.DataFrame, level: float, rng: random.Random) -> pd.DataFrame:
    out = df.copy()
    usd = pd.to_numeric(out["usd_amount_sum"], errors="coerce").fillna(0.0)
    for i in out.index:
        usd.loc[i] = usd.loc[i] * rng.uniform(1.0 - level, 1.0 + level)
    out["usd_amount_sum"] = usd.map(lambda x: f"{x:.12g}")
    return out


def _add_decoys(eth, bnb, lab, hints, *, mult: int, rng: random.Random) -> tuple:
    # add (mult-1) extra decoy pairs per template by cloning existing decoy rows with new offsets
    eth_n = eth[eth["flow_id"].astype(str).str.contains("__synth_noise_src")]
    bnb_n = bnb[bnb["flow_id"].astype(str).str.contains("__synth_noise_dst")]
    eth_new, bnb_new, lab_new = [], [], []
    noise_pairs = list(hints.get("eval_hints", {}).get("noise_decoy_pairs") or [])
    for extra in range(1, mult):
        off = 120 + 60 * extra
        for _, r in eth_n.iterrows():
            rid = str(r["flow_id"]).replace("__synth_noise_src_", f"__synth_noise_srcx{extra}_")
            row = r.to_dict()
            row["flow_id"] = rid
            row["tx_hashes"] = str(row["tx_hashes"]) + f"x{extra}"
            row["route_id"] = rid
            eth_new.append(row)
        for _, r in bnb_n.iterrows():
            rid = str(r["flow_id"]).replace("__synth_noise_dst_", f"__synth_noise_dstx{extra}_")
            row = r.to_dict()
            row["flow_id"] = rid
            row["tx_hashes"] = str(row["tx_hashes"]) + f"x{extra}"
            row["route_id"] = rid
            row["start_time"] = str(int(round(float(row["start_time"]) + off)))
            row["end_time"] = str(int(round(float(row["end_time"]) + off)))
            bnb_new.append(row)
            # matching label noise row
            src_id = rid.replace("__synth_noise_dstx", "__synth_noise_srcx")
            lab_new.append({
                "src_flow_id": src_id, "dst_flow_id": rid, "pattern_type": "one_to_one",
                "label_source": "semi_synthetic_delay_noise", "label_confidence": "0.24",
                "median_delay_sec": str(int(float(r["start_time"]) - 0)), "dst_amount_usd": r["usd_amount_sum"],
            })
            noise_pairs.append([src_id, rid])
    eth = pd.concat([eth, pd.DataFrame(eth_new)], ignore_index=True)
    bnb = pd.concat([bnb, pd.DataFrame(bnb_new)], ignore_index=True)
    lab = pd.concat([lab, pd.DataFrame(lab_new)], ignore_index=True)
    hints = json.loads(json.dumps(hints))
    hints["eval_hints"]["noise_decoy_pairs"] = noise_pairs
    return eth, bnb, lab, hints


def _add_unmatched(eth, bnb, lab, hints, *, mult: int, rng: random.Random) -> tuple:
    eth_u = eth[eth["flow_id"].astype(str).str.contains("__synth_unmatched_src")]
    bnb_h = bnb[bnb["flow_id"].astype(str).str.contains("__synth_hidden_dst")]
    eth_new, bnb_new, lab_new = [], [], []
    truth_unmatched = list(hints.get("eval_hints", {}).get("truth_unmatched_src_flows") or [])
    for extra in range(1, mult):
        for _, r in eth_u.iterrows():
            rid = str(r["flow_id"]).replace("__synth_unmatched_src", f"__synth_unmatchedx{extra}_src")
            row = r.to_dict()
            row["flow_id"] = rid
            row["tx_hashes"] = str(row["tx_hashes"]) + f"u{extra}"
            row["route_id"] = rid
            eth_new.append(row)
        for _, r in bnb_h.iterrows():
            rid = str(r["flow_id"]).replace("__synth_hidden_dst", f"__synth_hiddenx{extra}_dst")
            row = r.to_dict()
            row["flow_id"] = rid
            row["tx_hashes"] = str(row["tx_hashes"]) + f"u{extra}"
            row["route_id"] = rid
            bnb_new.append(row)
            src_id = rid.replace("__synth_hiddenx", "__synth_unmatchedx").replace("_dst", "_src")
            lab_new.append({
                "src_flow_id": src_id, "dst_flow_id": rid, "pattern_type": "one_to_one",
                "label_source": "semi_synthetic_unmatched", "label_confidence": "0.2",
                "dst_amount_usd": "0", "matched_dst_amount_usd": "0", "flow_mass_ratio_dst": "0",
            })
            truth_unmatched.append(src_id)
    eth = pd.concat([eth, pd.DataFrame(eth_new)], ignore_index=True)
    bnb = pd.concat([bnb, pd.DataFrame(bnb_new)], ignore_index=True)
    lab = pd.concat([lab, pd.DataFrame(lab_new)], ignore_index=True)
    hints = json.loads(json.dumps(hints))
    hints["eval_hints"]["truth_unmatched_src_flows"] = truth_unmatched
    return eth, bnb, lab, hints


# ---------------------------------------------------------------- run
def run_variant(
    variant_name: str,
    *,
    bridges: tuple[str, ...] = BRIDGES,
    seeds: tuple[int, ...] = SEEDS,
    variant: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    reg: float = 0.05,
    reg_m: float = 0.5,
) -> list[dict[str, Any]]:
    ensure_inputs()
    rows: list[dict[str, Any]] = []
    for br in bridges:
        for seed in seeds:
            out_dir = AUDIT / "variants" / variant_name / br / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            eth, bnb, lab, hints = apply_variant(br, seed, variant or {})
            eth.to_csv(out_dir / "flow_segments_eth_synth.csv", index=False)
            bnb.to_csv(out_dir / "flow_segments_bnb_synth.csv", index=False)
            (out_dir / "labels").mkdir(exist_ok=True)
            lab.to_csv(out_dir / "labels" / "synthetic_flow_labels.csv", index=False)
            (out_dir / "labels" / "synthetic_uot_eval_metrics.json").write_text(
                json.dumps(hints, indent=2) + "\n", encoding="utf-8")
            run_standalone_flow_uot(
                out_dir,
                src_flows_csv=out_dir / "flow_segments_eth_synth.csv",
                dst_flows_csv=out_dir / "flow_segments_bnb_synth.csv",
                flow_labels_csv=out_dir / "labels" / "synthetic_flow_labels.csv",
                uot_reg=float(reg),
                uot_reg_m=float(reg_m),
                uot_decode_threshold=1e-9,
                uot_cost_weights=weights,
                uot_backend="pot",
                uot_max_delay_sec=21600.0,
                uot_causal_violation_penalty=5.0,
                uot_lambda_risk=0.25,
                uot_causal_infeasible_delay_sec=None,
                uot_export_matrix=False,
                uot_export_cost_components=False,
                uot_export_cost_matrix_csv=False,
                uot_allow_unmatched=True,
                uot_use_graph_embedding=False,
                graph_ranker_checkpoint=None,
                uot_ablation="none",
                run_flow_baselines=False,
                flow_label_min_confidence=0.0,
                synthetic_eval_hints_path=out_dir / "labels" / "synthetic_uot_eval_metrics.json",
                uot_flow_dst_top_k=200,
                uot_flow_max_matrix_cells=6_000_000,
                uot_pool_strategy="default",
            )
            ev = json.loads((out_dir / "eval" / "uot_evaluation_metrics.json").read_text(encoding="utf-8"))
            rows.append({
                "variant": variant_name, "bridge": br, "seed": seed,
                "split_recovery": ev.get("split_recovery_rate"),
                "merge_recovery": ev.get("merge_recovery_rate"),
                "num_predicted_edges": ev.get("num_predicted_edges"),
                "tp": ev.get("num_true_positive_edges"),
                "truth": ev.get("num_true_edges"),
            })
            print(f"[{variant_name}] {br} seed {seed}: split={rows[-1]['split_recovery']} merge={rows[-1]['merge_recovery']}", flush=True)
    return rows


if __name__ == "__main__":
    # sanity: reproduce the main run exactly (no overrides) for Celer seed 42
    rows = run_variant("repro_check", bridges=("Celer",), seeds=(42,), variant={})
    print(rows)
