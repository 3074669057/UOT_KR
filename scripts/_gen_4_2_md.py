import json, csv
from pathlib import Path
from datetime import datetime, timezone

OUT = Path("out/rc_uot_v2_2")
AL = OUT / "acquisition_launch"
AL.mkdir(parents=True, exist_ok=True)
ts = datetime.now(timezone.utc).isoformat()

# batch_plan.md
(AL / "batch_plan.md").write_text("""# Batch Acquisition Plan (v4.2)

| Batch | Name | Groups | Expected Nontrivial | Purpose | Scientific | Confirmation |
|-------|------|--------|--------------------|---------|-----------|-------------|
| 0 | Schema pilot | 20 | 3 | Schema and ingestion test only | No | No |
| 1 | Feasibility | 400 | 60 | Annotation feasibility and yield | No | No |
| 2 | Development | 1,000 | 150 | Reach dev signal gate | Yes | No |
| 3 | Calibration | 500 | 75 | Calibration quota | Yes | No |
| 4 | Confirmation reserve | 1,700 | 255 | Confirmation quota | Yes | Yes |

## Quota Tracking
| Cohort | Min Nontrivial | Target Nontrivial |
|--------|---------------|-------------------|
| Development | 150 | 180 |
| Calibration | 50 | 60 |
| Confirmation | 197 | 220 |

## Dynamic Updates
Batch sizes may be updated based on observed yield after each batch labels are frozen.
Confirmation quota (197 minimum) must NEVER be lowered.

## Confirmation Isolation
Sealed and isolated after allocation. No solver access before readiness gate.
""", encoding="utf-8")
print("6/10 batch_plan.md")

# cohort_allocation_protocol.md
(AL / "cohort_allocation_protocol.md").write_text("""# Cohort Allocation Protocol (v4.2)

## Quota-Based Allocation (NOT percentage-based)

| Cohort | Nontrivial Min | Nontrivial Target |
|--------|---------------|-------------------|
| Development | 150 | 180 |
| Calibration | 50 | 60 |
| Confirmation | 197 | 220 |
| **Total** | **397** | **460** |

## Allocation Rules
1. Each independence group belongs to exactly one cohort
2. Allocation is frozen before any solver prediction
3. Allowed inputs: independence_group_id, ambiguity stratum, frozen topology label, annotation validity
4. Forbidden inputs: solver score, solver correctness, RC-UOT performance, baseline performance, q value, confirmation metric

## Why Not 60/20/20?
A 60/20/20 percentage split cannot simultaneously deliver 150 dev + 50 cal + 197 conf nontrivial groups.
The percentage approach was replaced with absolute quotas.

## Two Independent Manifests
- mechanism_enriched_cohort_manifest.jsonl: For solver performance conditional on nontrivial candidates
- population_natural_cohort_manifest.jsonl: For natural prevalence estimation with IPW

## Confirmation Cohort Seal
Once allocated, confirmation cohort is sealed (confirmation_cohort_seal.json, immutable).
No solver prediction, q fitting, threshold tuning, or debugging on confirmation data.
""", encoding="utf-8")
print("7/10 cohort_allocation_protocol.md")

# provider_sample_validation_spec.md
(AL / "provider_sample_validation_spec.md").write_text("""# Provider Sample Validation Specification (v4.2)

## Purpose
Before accepting a full dataset, require a 20-group unlabeled schema pilot.

## Validation Checks (label-free only)
1. Schema completeness
2. Timestamp parsing validity
3. Chain and asset field validity
4. Transaction identifier uniqueness
5. Candidate generation field availability
6. Zero duplicate records
7. Zero forbidden historical overlap
8. Independence-group constructability
9. Ambiguity feature computability

## Forbidden
- Solver performance computation
- q value calculation
- Any metric requiring gold labels

## Gate
Only when all checks pass can full data collection proceed.
""", encoding="utf-8")
print("8/10 provider_sample_validation_spec.md")

# acquisition_launch_checklist.md
(AL / "acquisition_launch_checklist.md").write_text("""# Acquisition Launch Checklist (v4.2)

## Pre-Launch (Complete)
- [x] Sample allocation corrected (quota-based)
- [x] Cohort allocation protocol frozen
- [x] Mechanism vs population cohorts separated
- [x] Provider sample validation spec defined
- [x] Batch acquisition plan generated
- [x] Collection forecast utility implemented
- [x] Confirmation isolation enforced
- [x] Evaluator sealed
- [x] Power analysis updated
- [x] Formal runbook updated

## Launch Gate (Pending)
- [ ] Provider submits schema pilot (20 groups)
- [ ] validate_provider_sample passes
- [ ] Batch 1 (feasibility) collected and labeled
- [ ] Observed yield computed

## Readiness Gate (Pending)
- [ ] Dev nontrivial >= 150
- [ ] Cal nontrivial >= 50
- [ ] Conf nontrivial >= 197
- [ ] Signal gate passed

## Status: READY_FOR_PROVIDER_SAMPLE
""", encoding="utf-8")
print("9/10 acquisition_launch_checklist.md")

# stage_4_2_report.md
(AL / "stage_4_2_report.md").write_text("""# RC-UOT-v2.2 Stage 4.2: Acquisition Allocation Correction and Launch Package

## Status: READY_FOR_PROVIDER_SAMPLE

## 1. Why 220 Total Nontrivial Was Insufficient
The 60/20/20 split cannot deliver 197 confirmation nontrivial groups.
220 * 0.30 = 66, far below the paired-power requirement for MDE=0.03.

## 2. Corrected Quotas
| Cohort | Min | Target |
|--------|-----|--------|
| Development | 150 | 180 |
| Calibration | 50 | 60 |
| Confirmation | 197 | 220 |

## 3. Collection Scale
- Expected nontrivial yield: 15%
- Raw groups: 460 / 0.15 = 3,067 + 15% buffer = 3,600

## 4. Mechanism vs Population Cohorts
- Mechanism: enriched, quota-based, for solver evaluation
- Population: natural probability, for prevalence estimation

## 5. Batch Plan
0: 20 (pilot), 1: 400 (feasibility), 2: 1,000 (dev), 3: 500 (cal), 4: 1,700 (conf)

## 6. Confirmation Isolation
sealed, immutable, no solver access before readiness

## 7. Updated Runbook
Phases A (unlabeled check), B (label+cohort), C (dev+cal), D (one-shot conf)

## 8. Tests Added (7 new)
quota, isolation, separateness tests added to e2e suite
""", encoding="utf-8")
print("10/10 stage_4_2_report.md")
print("All 10 deliverables complete")
