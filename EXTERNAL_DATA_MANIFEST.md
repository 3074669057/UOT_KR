# EXTERNAL_DATA_MANIFEST — what is **not** in this repository, and how to obtain it

This release is intentionally curated. Everything needed to *inspect* and *independently validate*
the confirmatory (R7) result is shipped. The items below are excluded because they are either
(a) large deterministic intermediates that the released code regenerates, or (b) restricted raw data.

**No DOI is claimed.** Nothing has been deposited to Zenodo / OSF. If a deposit is made, add the DOI
and the SHA256 of the deposited archive to section 3 of this file.

---

## 1. Regenerable large intermediates (safe to omit)

These are *outputs*, not primary data. Each is produced by released code from released inputs.

| item | origin path (in the private workspace) | size | why omitted | how to regenerate |
|---|---|---|---|---|
| Development-cell cost/transport arrays | `out/multi_bridge_expansion/cost_transport_diagnosis/plans/dev/<BRIDGE>/seed_<201..205>/` (`cost.npz`, `ids.npz`, `transport_uot.npz`, `transport_bot.npz`, `labels.csv`) | **132.7 MB**, 210 files | `.npz` arrays; deterministic function of the frozen generator + frozen cost weights | `python scripts/multi_bridge/run_build_dev_plans.py` (writes to the same path). **Requires the real flow-feature inputs — see section 2.** |
| Amount-free candidate development arrays | `out/multi_bridge_expansion/amount_free_candidate_dev/` | **721.7 MB**, 650 files | consumed by `run_r7_selection.py`; superseded by the frozen selected rule shipped here | `python scripts/multi_bridge/run_amount_free_dev.py`, then `python scripts/multi_bridge/run_af_analysis.py` |
| Conditional-plan holdout arrays | `out/multi_bridge_expansion/conditional_plan_holdout_results/` | **84.4 MB**, 281 files | frozen evidence trail for the earlier conditional-decoding line; its report is shipped in `paper/` | `python scripts/multi_bridge/run_conditional_plan_dev.py` |
| Decoder plan-quality audit arrays | `out/multi_bridge_expansion/decoder_plan_quality_audit/` | **124.8 MB**, 173 files | diagnostic intermediate | `python scripts/multi_bridge/run_plan_quality_audit.py` |
| R7 per-cell transport/cost matrices | `out/r7_confirmatory_kernel_ranking_20260917/{selection,confirmatory}/**/*.npz` | **≈643 MB**, 288 files | `.npz`; every *metric* derived from them is shipped in the JSON/CSV result files | `python scripts/run_r7_selection.py --all` then `python scripts/run_r7_confirmatory_kernel_ranking.py --execute` (the latter now refuses: the one-shot holdout is spent) |
| Paper-main pipeline transport plans | `out/paper_full_pipeline_run/**/*.npz`, `out/leave_anchor_out_real/**/*.npz` | up to **197 MB** per file | intermediates of the paper-main pipeline; only their summary metrics and figures are released | `python run.py` with the configuration in `config/defaults.json` (**requires the raw corpora — see section 2**) |
| Historical three-bridge variant scan | `out/multi_bridge_expansion/faithful_flow_structural_three_bridges_audit/` | **16.4 GB**, 9 425 files | exploratory variant sweep with no claim in the paper | not regenerated; superseded by the R7 confirmatory line |
| Chapter-4 packaging bundles | `out/reviewer_artifacts/chapter4_data_package.zip` and the `out/chapter4_*_package/` trees | **849 MB** (single file) | redundant repackaging of artifacts already present in `paper/` | `python scripts/build_chapter4_repro_packages.py` |

---

## 2. Restricted raw data (not released, by decision)

| item | why withheld |
|---|---|
| Bridge candidate / flow tables and provider exports used to build the real flow features (e.g. the per-bridge `Celer/Multi/Poly` candidate universes, provider API responses) | Contains **address-level** entities. The manuscript states that address-level entity labels are **not** released. |
| Address lists and token maps under `data/` | Address-level data; the released flow-level labels keep only the anonymised identifiers required for evaluation. |
| Provider RPC endpoints and API keys | Never released; see `SECURITY.md`. |

**Consequence, stated plainly:** the R5 and R6 experiment lines **cannot be re-executed** from this
repository alone. Their released inputs to the analysis and all released outputs are present, so every
reported number can be inspected and re-aggregated offline, but re-solving the UOT problems requires
the development-cell arrays, which in turn require the restricted raw corpora. This limitation is
repeated in `README.md` §3 and §6.

---

## 3. Integrity records for the omitted material

Where the private workspace recorded a hash of the original file, it is preserved inside this
repository in the shipped provenance manifests:

| manifest | location | what it pins |
|---|---|---|
| R11 package hash list | `out/handoff_r11_remaining6/SHA256SUMS.txt` | 1 008 entries over the R11 `core/` and `e1/` trees. **963 of them are shipped** (loose, or inside `R11_CORE_evidence.zip` / `R11_E1_evidence.zip`). The **45 `.npz` entries are excluded** — see below. |
| Frozen R7 protocol manifest | `out/r7_confirmatory_kernel_ranking_20260917/config/FROZEN_PROTOCOL_MANIFEST.json` | 42 hashed protocol files |
| R7 release manifest | `out/r7_confirmatory_kernel_ranking_20260917/MANIFEST.json` | 505 files with path, bytes, SHA256, role, stage and frozen status |
| Paper package provenance | `paper/ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/provenance/` | `ARCHIVE_INDEX.csv`, `EXCLUDED_LARGE_RAW_DATA.csv` (42 099 explicitly excluded large raw files), `PAPER_EXPERIMENTS_RESULTS_MANIFEST.json` |

### The 45 excluded `.npz` entries of `SHA256SUMS.txt`

15 files under `core/pending5_holdout_d4/` (28.9 MB) and 30 files under the `e1/` closure set
(57.5 MB). Each is a cell-input / cost / transport array of the R11 forensic evidence. They are
excluded under the same rule as every other `.npz` in this release. To verify the shipped subset of
`SHA256SUMS.txt` yourself:

```bash
python unpack_release.py            # expands the R11 evidence archives
# then, for every non-.npz line of SHA256SUMS.txt, the file must be present and match
```

---

## 4. Re-checking this manifest

```bash
python release_check.py --json release_check_report.json
python unpack_release.py --verify-only
```

Both commands are offline and must pass on a fresh clone.
