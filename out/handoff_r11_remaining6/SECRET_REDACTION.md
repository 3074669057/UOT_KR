# SECRET_REDACTION.md

R11 remaining-evidence forensic package — secret / privacy filter report.

## Result

**No secret was copied. No sanitized copy was required.**

`SANITIZED_COPY_REQUIRED = NO`
`REDACTED_FIELDS = 0`
`SECRET_VALUES_WRITTEN_TO_ANY_REPORT = 0`

## Filter applied

Every candidate file was matched against this deny pattern before copying (case-insensitive,
matched on the basename at any depth of the origin path):

```
(^|[\/])(\.env[^\/]*|local\.json|local\.[^\/]*\.json|secrets?|credentials?|cookies?|
          id_rsa[^\/]*|.*\.pem|.*\.key|.*\.p12|keystore[^\/]*)$
```

Zero files matched, therefore zero files were skipped by the filter and zero sanitized
copies were produced. The `SECRET_FILTER` reason count in `_tools/copy_manifest.json`
is **0**.

## Files deliberately NOT copied (known live-credential carriers, outside the plan)

| Origin path | Why excluded |
|---|---|
| `.env` | repository root environment file (RPC URL / API key carrier per `README.md` §2) |
| `config/local.json` | documented local override for `nodereal` / `ethereum` secrets; gitignored |

Both are ignored by the repository's own `.gitignore` and were never on the copy list for
any of the six pending items. They are recorded here so a reviewer can confirm the
exclusion was deliberate rather than an oversight.

## Privacy note carried from the manuscript

`full_manuscript_final.md` §5.10 states that address-level entity labels are **not**
released and that released flow-level labels keep only the anonymized identifiers needed
for evaluation, and that provider API keys are not released. This package copies frozen
artifacts as they already exist in the repository; no additional de-anonymization was
performed and none of the copied artifacts were transformed.

## Verification command

A reviewer can re-run the filter over the package:

```powershell
Get-ChildItem out\handoff_r11_remaining6 -Recurse -File |
  Where-Object { $_.Name -match '^\.env|^local\.json$|secret|credential|cookie|id_rsa|\.pem$|\.key$|\.p12$|keystore' } |
  Select-Object FullName
```

Expected output: empty.
