# Claim Boundary (Cross AML Tool)

This document defines what the open-source research prototype **may** and **may not** claim. It aligns with the parent paper's Phase 29.1 boundaries and does not modify frozen experiment gates.

## Allowed claims

- **Evidence-qualified AML investigation leads** — outputs prioritize auditable hypotheses for human triage.
- **Coverage-qualified matching** — high-confidence paths only when Tier A/B event evidence gates pass.
- **Interpretable correspondence hypotheses** — reports include features, evidence refs, tiers, and explanations.
- **Abstention on unsupported canonical edges** — `abstain` when coverage is Uncovered or evidence is insufficient.
- **Precision/F1-oriented triage** (research context) — standalone scorer inspired by RC-UOT-Q; not the Phase 29.1 trained checkpoint.

## Forbidden claims

Do **not** describe this tool or its reports as:

| Forbidden | Reason |
|-----------|--------|
| **Full-scope high P/R** | `full_scope_claim` is false by design |
| **Universal superiority** over ABCTracer, Connector, or any method | Paper gates: precision/F1 Pareto only |
| **Legal proof** or conviction evidence | Outputs are investigation leads, not court-ready proof |
| **Guaranteed candidate discovery** | Standard RPC is limited; indexer/archive/trace backend required at scale |
| **High-confidence match without event evidence** | Uncovered / low evidence → abstain or low-confidence only |
| **"ABCTracer failed"** or **"Connector failed"** | External baselines remain strong references |
| **Production AML product** or **compliance certification** | Research prototype only |

## Report semantics

| Field / decision | Meaning |
|------------------|---------|
| `match` | Strongest automated hypothesis; **requires investigator review** |
| `low_confidence_candidate` | Weak automated support; not sole escalation basis |
| `diagnostic_candidate` | Tier C diagnostic only |
| `abstain` | Tool declines to assert correspondence |
| `claim_boundary.full_scope_claim` | Always `false` in reference implementation |
| `candidate_discovery_limited_without_indexer` | `true` when discovery runs without dest txs and no search backend |

## Coverage tiers

| Tier | Meaning |
|------|---------|
| **A** | Strong bridge / transfer-key alignment |
| **B** | Strong amount-time-token consistency |
| **C** | Diagnostic only — weak evidence |
| **Uncovered** | Abstain — do not assert correspondence |

## Investigator actions

| Decision | Suggested action |
|----------|------------------|
| `match` | Review promptly; corroborate bridge logs and addresses |
| `low_confidence_candidate` | Secondary review queue |
| `diagnostic_candidate` | Weak lead only |
| `abstain` | Do not escalate on this pair alone |
| `non_match` | Deprioritize unless new evidence appears |

See also [DISCLAIMER.md](DISCLAIMER.md) and [SECURITY.md](SECURITY.md).
