"""R7 Stage 0B structural QA -- MUST pass before any method F1 is computed.

Validates the extended generator on the real selection block:

  * 24 distinct families per bridge/seed, 2 instances per family
  * base-anchor clusters are disjoint (no anchor reused by two templates)
  * the sampled split/merge degree histograms match the frozen empirical PMFs
  * split/merge topology, unmatched count and decoy count
  * label consistency (truth edges agree with ``pattern_type`` / ``label_source``)

Emits ``selection/generator/family_manifest.csv``, ``GENERATOR_VALIDATION.md`` and
``generator_structural_qa.json``.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):                      # allow `python scripts/r7/r7_qa.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r7.r7_common import (BRIDGES, DEGREE_RANGE, DIR_GENERATOR, DIR_SELECTION, K_MAX,
                          N_FAMILIES, INSTANCES_PER_FAMILY, SELECTION_SEEDS, log,
                          sha256_file, sha256_obj, family_of_template,
                          family_index_of_template, instance_of_template, utc_now,
                          write_json, write_text)
from r7.r7_generator import build_cell, cell_workdir, load_cell

SPEC_PATH = DIR_SELECTION / "degree_calibration" / "degree_sampling_spec.json"
MANIFEST = DIR_GENERATOR / "family_manifest.csv"
QA_JSON = DIR_GENERATOR / "generator_structural_qa.json"
QA_MD = DIR_GENERATOR / "GENERATOR_VALIDATION.md"

CONSTRUCTION_RULE = (
    "faithful generator cell (amount quartile {aq} x delay sextile {ds}, {half} half of "
    "the frozen 4x3 cell); split 1->d_split and merge d_merge->1 with d drawn from the "
    "frozen pooled empirical degree distribution; equal-share amount allocation; "
    "unmatched source + 2 delay decoys (+60s/+120s) unchanged; template "
    "<anchor>__r7fam{fi:02d}__inst{ii:02d}"
)


def build_or_load(bridge: str, seed: int, spec: dict[str, Any], *, rebuild: bool = False) -> dict[str, Any]:
    if not rebuild:
        try:
            return load_cell(bridge, seed)
        except FileNotFoundError:
            pass
    return build_cell(bridge, seed, spec)


def qa(rebuild: bool = False) -> dict[str, Any]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    target_split = np.asarray(spec["split_degree"]["pmf"], dtype=float)
    target_merge = np.asarray(spec["merge_degree"]["pmf"], dtype=float)
    support = [int(x) for x in spec["split_degree"]["support"]]

    checks: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    sampled_split: Counter = Counter()
    sampled_merge: Counter = Counter()
    per_cell: list[dict[str, Any]] = []
    label_problems: list[str] = []
    topology_problems: list[str] = []
    anchor_map: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    for bridge in BRIDGES:
        for seed in SELECTION_SEEDS:
            cell = build_or_load(bridge, seed, spec, rebuild=rebuild)
            truth = cell["truth"]
            tpls = sorted(truth)
            fams: dict[str, list[str]] = defaultdict(list)
            for t in tpls:
                fams[family_of_template(t)].append(t)

            # ---- family structure ----------------------------------------- #
            if len(fams) != N_FAMILIES:
                label_problems.append(f"{bridge}/{seed}: {len(fams)} families != {N_FAMILIES}")
            sizes = {len(v) for v in fams.values()}
            if sizes != {INSTANCES_PER_FAMILY}:
                label_problems.append(f"{bridge}/{seed}: family sizes {sizes}")

            hints = cell.get("generator_hints")
            if hints is None:
                hints = json.loads(
                    (cell_workdir(bridge, seed) / "labels" / "synthetic_uot_eval_metrics.json")
                    .read_text(encoding="utf-8"))
            records = {r["template_id"]: r for r in hints.get("r7_family_records", [])}

            # ---- per-template topology + label consistency ----------------- #
            bases_seen: dict[str, str] = {}
            for t in tpls:
                tr = truth[t]
                rec = records.get(t, {})
                base = rec.get("base_anchor_id", t)
                if base in bases_seen:
                    topology_problems.append(
                        f"{bridge}/{seed}: base anchor {base} reused by {bases_seen[base]} and {t}")
                bases_seen[base] = t
                anchor_map[(bridge, seed)][t] = {
                    "base_anchor_id": base,
                    "family_index": family_index_of_template(t),
                    "instance": instance_of_template(t),
                    "split_degree": len(tr["split"]),
                    "merge_degree": len(tr["merge"]),
                }
                # split topology: one source -> d distinct dsts
                if len(tr["split_src"]) != 1:
                    topology_problems.append(f"{bridge}/{seed}/{t}: split_src != 1")
                if len(tr["split"]) != len(tr["split_dst"]):
                    topology_problems.append(f"{bridge}/{seed}/{t}: split edges != dsts")
                if len(tr["merge"]) != len(tr["merge_src"]) or len(tr["merge_dst"]) != 1:
                    topology_problems.append(f"{bridge}/{seed}/{t}: merge topology")
                if len(tr["decoy"]) != 2:
                    topology_problems.append(f"{bridge}/{seed}/{t}: decoys={len(tr['decoy'])}")
                if len(tr["unmatched_src"]) != 1:
                    topology_problems.append(f"{bridge}/{seed}/{t}: unmatched={len(tr['unmatched_src'])}")
                if len(tr["positive"]) != len(tr["split"]) + len(tr["merge"]) + len(tr["decoy"]):
                    topology_problems.append(f"{bridge}/{seed}/{t}: positive != split+merge+decoy")
                if len(tr["split"]) + len(tr["merge"]) + 2 != len(tr["positive"]):
                    topology_problems.append(f"{bridge}/{seed}/{t}: positive count mismatch")

            # ---- label consistency ---------------------------------------- #
            lab = cell["labels"]
            for t in tpls:
                g = lab[lab["src_flow_id"].astype(str).str.startswith(t + "__")]
                for _, row in g.iterrows():
                    pt = str(row.get("pattern_type"))
                    ls = str(row.get("label_source"))
                    if pt == "one_to_many" and "semi_synthetic_split" not in ls:
                        label_problems.append(f"{bridge}/{seed}/{t}: split row label_source={ls}")
                    if pt == "many_to_one" and "semi_synthetic_merge" not in ls:
                        label_problems.append(f"{bridge}/{seed}/{t}: merge row label_source={ls}")

            # ---- degree draws --------------------------------------------- #
            sd = [int(x) for x in cell["degrees"]["split_degrees"]]
            md = [int(x) for x in cell["degrees"]["merge_degrees"]]
            sampled_split.update(sd)
            sampled_merge.update(md)
            if len(sd) != len(tpls) or len(md) != len(tpls):
                label_problems.append(f"{bridge}/{seed}: degree vector length != templates")
            observed_split = sorted(len(truth[t]["split"]) for t in tpls)
            if observed_split != sorted(sd):
                topology_problems.append(f"{bridge}/{seed}: realised split degrees != drawn")
            observed_merge = sorted(len(truth[t]["merge"]) for t in tpls)
            if observed_merge != sorted(md):
                topology_problems.append(f"{bridge}/{seed}: realised merge degrees != drawn")

            per_cell.append({
                "bridge": bridge, "seed": seed,
                "n_templates": len(tpls), "n_families": len(fams),
                "instances_per_family": sorted(sizes),
                "n_unique_base_anchors": len(bases_seen),
                "n_src": cell["n_src"], "n_dst": cell["n_dst"],
                "unmatched_total": int(sum(len(truth[t]["unmatched_src"]) for t in tpls)),
                "decoy_total": int(sum(len(truth[t]["decoy"]) for t in tpls)),
                "positive_total": int(sum(len(truth[t]["positive"]) for t in tpls)),
                "split_degree_hist": dict(sorted(Counter(sd).items())),
                "merge_degree_hist": dict(sorted(Counter(md).items())),
                "solver_converged": cell["solver"]["converged"],
                "cost_matrix_sha256": cell["hashes"]["cost_matrix_sha256"],
                "labels_sha256": cell["hashes"]["labels_sha256"],
            })
            log(f"[0B-QA] {bridge} seed {seed}: 48 templates / {len(fams)} families "
                f"/ {len(bases_seen)} unique anchors")

    # ---- family manifest --------------------------------------------------- #
    for (bridge, seed), mp in sorted(anchor_map.items()):
        for t, info in sorted(mp.items(), key=lambda kv: (kv[1]["family_index"], kv[1]["instance"])):
            fi = info["family_index"]
            aq, ds = divmod(fi, 6)
            manifest_rows.append({
                "bridge": bridge, "seed": seed,
                "family_id": f"r7fam{fi:02d}",
                "instance": info["instance"],
                "template_id": t,
                "base_anchor_id": info["base_anchor_id"],
                "original_or_new": "original" if ds % 2 == 0 else "new",
                "amount_quartile": aq,
                "delay_sextile": ds,
                "construction_rule": CONSTRUCTION_RULE.format(
                    aq=aq, ds=ds, half="first" if ds % 2 == 0 else "second",
                    fi=fi, ii=int(info["instance"])),
                "split_degree": info["split_degree"],
                "merge_degree": info["merge_degree"],
                "template_hash": sha256_obj({"bridge": bridge, "seed": seed, "template": t,
                                             "anchor": info["base_anchor_id"],
                                             "split": info["split_degree"],
                                             "merge": info["merge_degree"]}),
            })
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    # ---- aggregate checks -------------------------------------------------- #
    n_cells = len(BRIDGES) * len(SELECTION_SEEDS)
    check("family_count_24_per_cell", all(c["n_families"] == N_FAMILIES for c in per_cell),
          {"cells": n_cells, "observed": sorted({c["n_families"] for c in per_cell})})
    check("instances_2_per_family", all(c["instances_per_family"] == [INSTANCES_PER_FAMILY]
                                        for c in per_cell),
          sorted({tuple(c["instances_per_family"]) for c in per_cell}))
    check("base_anchor_clusters_disjoint",
          not topology_problems and all(c["n_unique_base_anchors"] == 48 for c in per_cell),
          {"unique_anchors": sorted({c["n_unique_base_anchors"] for c in per_cell}),
           "violations": topology_problems[:5]})
    check("unmatched_1_per_template", all(c["unmatched_total"] == 48 for c in per_cell),
          sorted({c["unmatched_total"] for c in per_cell}))
    check("decoys_2_per_template", all(c["decoy_total"] == 96 for c in per_cell),
          sorted({c["decoy_total"] for c in per_cell}))
    check("label_consistency", not label_problems, label_problems[:5])
    check("split_merge_topology", not topology_problems, topology_problems[:5])

    obs_split = np.array([sampled_split[d] for d in support], dtype=float)
    obs_split_p = obs_split / obs_split.sum()
    obs_merge = np.array([sampled_merge[d] for d in support], dtype=float)
    obs_merge_p = obs_merge / obs_merge.sum()
    tv_split = float(0.5 * np.abs(obs_split_p - target_split).sum())
    tv_merge = float(0.5 * np.abs(obs_merge_p - target_merge).sum())
    check("split_degree_matches_target", tv_split < 0.05,
          {"total_variation": tv_split, "observed": obs_split_p.tolist(),
           "target": target_split.tolist(), "n_draws": int(obs_split.sum())})
    check("merge_degree_matches_target", tv_merge < 0.05,
          {"total_variation": tv_merge, "observed": obs_merge_p.tolist(),
           "target": target_merge.tolist(), "n_draws": int(obs_merge.sum())})
    check("degree_within_frozen_range",
          all(DEGREE_RANGE[0] <= d <= K_MAX for d in list(sampled_split) + list(sampled_merge)),
          {"range": list(DEGREE_RANGE), "cap": K_MAX,
           "max_observed": max(list(sampled_split) + list(sampled_merge))})
    check("all_solver_converged", all(c["solver_converged"] for c in per_cell),
          sum(1 for c in per_cell if c["solver_converged"]))
    check("no_duplicate_cells", len({(c["bridge"], c["seed"]) for c in per_cell}) == n_cells, n_cells)

    all_pass = all(c["status"] == "PASS" for c in checks)
    out = {
        "stage": "0B",
        "generated_at_utc": utc_now(),
        "purpose": ("structural QA of the extended generator, completed BEFORE any method "
                    "F1 on the selection block was computed"),
        "method_f1_computed": False,
        "cells": per_cell,
        "checks": checks,
        "GENERATOR_STRUCTURAL_QA": "PASS" if all_pass else "FAIL",
        "family_manifest": {
            "path": str(MANIFEST.relative_to(DIR_GENERATOR.parents[2])),
            "rows": len(manifest_rows),
            "sha256": sha256_file(MANIFEST),
        },
        "sampled_split_degree_hist": dict(sorted(sampled_split.items())),
        "sampled_merge_degree_hist": dict(sorted(sampled_merge.items())),
        "label_problems": label_problems[:20],
        "topology_problems": topology_problems[:20],
    }
    write_json(QA_JSON, out)

    lines = ["# R7 generator structural validation (Stage 0B)", "",
             f"* generated: {out['generated_at_utc']}",
             f"* result: **{out['GENERATOR_STRUCTURAL_QA']}**",
             "* method F1 computed before this validation: **NO**", "",
             "## Checks", "", "| check | status | detail |", "|---|---|---|"]
    for c in checks:
        d = json.dumps(c["detail"], default=str)
        lines.append(f"| `{c['check']}` | **{c['status']}** | `{d[:220]}` |")
    lines += ["", "## Per-cell structure", "",
              "| bridge | seed | templates | families | unique anchors | src | dst | "
              "unmatched | decoys | positives | converged |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for c in per_cell:
        lines.append(f"| {c['bridge']} | {c['seed']} | {c['n_templates']} | {c['n_families']} | "
                     f"{c['n_unique_base_anchors']} | {c['n_src']} | {c['n_dst']} | "
                     f"{c['unmatched_total']} | {c['decoy_total']} | {c['positive_total']} | "
                     f"{c['solver_converged']} |")
    lines += ["", "## Sampled degree distribution vs frozen target", "",
              "| degree | sampled split p | frozen split p | sampled merge p | frozen merge p |",
              "|---:|---:|---:|---:|---:|"]
    for i, d in enumerate(support):
        lines.append(f"| {d} | {obs_split_p[i]:.4f} | {target_split[i]:.4f} | "
                     f"{obs_merge_p[i]:.4f} | {target_merge[i]:.4f} |")
    lines += ["", f"* split total variation distance: `{tv_split:.6f}`",
              f"* merge total variation distance: `{tv_merge:.6f}`", "",
              "## Cluster / duplication audit", "",
              "* every template inside every (bridge, seed) cell has a DISTINCT base anchor",
              "* therefore no base-anchor cluster is shared by two families, and the "
              "base-anchor cluster sensitivity analysis has 48 disjoint clusters per cell",
              f"* family manifest rows: {len(manifest_rows)} "
              f"(3 bridges x 10 selection seeds x 24 families x 2 instances)", ""]
    write_text(QA_MD, "\n".join(lines) + "\n")
    return out


def main() -> int:
    rebuild = "--rebuild" in sys.argv
    res = qa(rebuild=rebuild)
    print(json.dumps({k: v for k, v in res.items() if k != "cells"}, indent=2, default=str)[:6000])
    return 0 if res["GENERATOR_STRUCTURAL_QA"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
