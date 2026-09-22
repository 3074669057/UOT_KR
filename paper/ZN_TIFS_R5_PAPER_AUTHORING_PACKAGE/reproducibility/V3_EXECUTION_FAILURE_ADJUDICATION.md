# V3_EXECUTION_FAILURE_ADJUDICATION.md

Human adjudication record for the v3 temporal external execution. Date:
2026-09-03.

## Adjudication

- v3 status = **EXECUTION_FAILURE** (permanent).
- Real execution attempts = **1** (prediction path entered once).
- Valid performance = **NONE**.
- Rerun = **NO** (forbidden, now and permanently).
- v3 is archived as: **PRE-REGISTERED LEVEL-I EXTERNAL EXECUTION FAILURE**.

## Allowed conclusion (frozen wording)

"The preregistered external execution failed before valid method performance
could be obtained."

## Forbidden conclusions (frozen)

- decoder succeeded externally
- decoder failed externally
- raw/conditional comparison result
- UOT/BOT external comparison
- external fan-out performance

None of these has a valid estimate and none may be written anywhere.

## Failure classification

- Failure A (missing `itertools` import in the inference step): **SOFTWARE
  IMPLEMENTATION DEFECT** — not a scientific-method modification.
- Failure B (zero marginal values → POT divisions by zero → NaN transport
  plan): **NUMERICAL DOMAIN / SOLVER INTERFACE DEFECT** — the solver wrapper
  implicitly assumed strictly positive marginals. The frozen missing-data
  policy (unpriceable flow → zero USD mass) is NOT reclassified as a policy
  failure; the defect is the wrapper's zero-handling.

## Next steps (human-authorized)

Failure adjudication + software-only repair (support-aware solver wrapper,
validated on non-v4 data) + new v4 external design. NO new external method
execution. v3 (b01–b03) and b04–b06 are permanently retired for performance
evaluation and must not become repair test data.
