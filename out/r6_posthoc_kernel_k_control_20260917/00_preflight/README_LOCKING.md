# Locking procedure

1. `--snapshot` recorded git HEAD, the pre-existing dirty status, the pre-existing
   relevant diff and the software versions into `00_preflight/`.
2. `--lock-spec` wrote `config/locked_spec.json` and its SHA256 into
   `config/locked_spec.sha256`. Both happened BEFORE any R6 cell was executed.
3. The specification is never edited in place. `load_spec()` re-hashes the file on
   every entry point and aborts on any mismatch.
4. If the specification had needed a correction, a NEW versioned file would have
   been created with the reason recorded here. No such correction was needed: the
   only pre-run adjustment was the switch from the author's proposed rules to the
   operational rules, and that switch is recorded inside the locked spec itself
   (`author_proposed_decision_rules` is preserved verbatim next to
   `operational_decision_rules`), made before any R6 result existed.
