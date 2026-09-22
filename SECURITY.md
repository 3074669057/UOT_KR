# SECURITY — secrets, credentials and data restrictions in this release

## 1. Pre-publication secret scan: result

The scan was run on the **final** file tree immediately before publication (not on an earlier
revision), using two independent passes plus a targeted search for every credential value known to
exist in the private workspace.

| pattern class | result |
|---|---|
| Provider RPC URLs with embedded keys (`nodereal.io/v1/<key>`, `infura.io/v3/<key>`, `alchemy.com/v2/<key>`, …) | **0 hits** |
| `sk-…`, `ghp_…`/`github_pat_…`, `AKIA…`, `AIza…`, `xox…`, `Bearer …` tokens | **0 hits** |
| PEM private keys (`-----BEGIN … PRIVATE KEY-----`), `id_rsa*`, `*.pem`, `*.key`, `*.p12` | **0 hits** |
| `api_key`/`secret`/`password`/`token` assigned a ≥16-character high-entropy literal | **0 real hits** (see §1.2) |
| `.env`, `config/local*.json`, `secrets/`, `credentials/` | **0 files present** |
| Authors' local absolute paths (`D:\trae\…`, `C:\Users\…`) | **0 residual**; 47 files normalised to `<REPO>` — see `PATH_SANITIZATION.md` |
| Targeted search for the 5 credential values known to exist in the private workspace | **0 occurrences** in any of the 2 085 files |

### 1.1 Three credential files were found and removed before publication

The curated staging step copied `config/` verbatim, which pulled in **three credential-bearing
files** from the private workspace. All three were identified by the scan, removed from the release,
and are now covered by `.gitignore`. They were **never committed and never pushed**.

| file (removed) | content | status |
|---|---|---|
| `config/local.json` | 3 × NodeReal API key | removed; replaced by `config/local.example.json` |
| `config/local.runtime.json` | the same 3 NodeReal keys + RPC window overrides | removed |
| `config/api-key-cross.json` | provider `key` + `secret` pair | removed |

Verification that nothing remains — every one of the 5 values was searched across **all 2 085 files**
of the release tree (text and binary): **0 hits**. The commands are in §4.

> **Action required by the authors (outside this repository).** Because these values existed in the
> private working tree and in the Gitee repository's ignored-but-present files, they should be
> **rotated at the provider** regardless of this release. Deleting them here does not invalidate a
> key that may have been exposed elsewhere.

### 1.2 The one benign "credential-shaped" match

`tests/test_ec_uot_q_final_protocol.py` contains
`TOKEN = "55d398326f99059ff775485246999027b3197955"` — the **public USDT-BSC token contract
address**, used as test input. Public chain data, not a secret.

The 55 remaining hits of the broad JSON-key heuristic are all the word `"token"` used as a *token
symbol* field (e.g. `"token": "0x…"`, `"token": tok`) or empty arrays (`"api_keys": []`). No file
contains a `"secret"` key with a literal value.

## 2. What was deliberately excluded

* `.env` (root environment file — RPC URL / API key carrier) and the `config/local*.json` overrides.
  Both are git-ignored and were never part of the released file set.
* `config/api-key-cross.json` (provider key/secret pair).
* The R11 evidence package's own `SECRET_REDACTION.md` records that its copy filter matched zero
  files and that `SANITIZED_COPY_REQUIRED = NO`; `local.json` and `.env` are listed there as
  deliberate exclusions.
* `scripts/_archive/` (43 one-off patch scripts that hard-coded the authors' workspace root) —
  removed as part of path sanitisation.

## 3. Data restrictions

* **Address-level entity labels are not released.**
* Released flow-level labels keep only the anonymised identifiers required for evaluation.
* **No provider API key is released.**
* Raw on-chain corpora are not released; see `EXTERNAL_DATA_MANIFEST.md` §2.

## 4. Verifying these claims yourself

```bash
# (a) no credential-bearing files by name
git ls-files | grep -Ei '(^|/)\.env|local\.json$|local\..*\.json$|api-key|secrets?/|credentials?/|id_rsa|\.pem$|\.key$|\.p12$'

# (b) no internal absolute paths
grep -rInE 'D:\\trae|C:\\Users\\' --include='*.py' --include='*.json' --include='*.md' . | head

# (c) no credential-shaped content (broad scan)
python release_check.py --json release_check_report.json
```

(a) and (b) are expected to produce no output.

## 5. Reporting

If you believe you have found a credential, a data-protection issue, or restricted data in this
repository, please open a GitHub issue marked **[security]** (without quoting the value) so it can
be removed.
