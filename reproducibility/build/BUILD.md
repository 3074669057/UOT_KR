# BUILD — how this release was assembled

This documents the curation pipeline that produced the repository you are reading, so the
selection is auditable rather than a black box. It is **not** needed to use the artifact; it
is here so a reviewer can see exactly what was kept, what was dropped, and how the integrity
manifests were produced.

The source private workspace is **not** part of this release and is not required by anything
below except step 1, which has already been run.

---

## Pipeline

| step | script | what it does |
|---|---|---|
| 1 | *(one-off, not shipped)* | Copy the approved scope out of the private workspace into a clean staging tree: `src/`, `scripts/`, `tests/`, `config/`, `schemas/`, `docs/`, `paper/ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE/`, `out/r5_*`, `out/r6_*`, `out/r7_*`, `out/handoff_r11_remaining6/`, `_risk_audit_tmp/` → `audit/risk_field/`, plus `ZN_TIFS_CN_R11_SUBMISSION_READY.docx` and its `SHA256SUMS.txt`. Excludes `__pycache__`, `*.pyc`, and `.npz`. |
| 1b | *(one-off, not shipped)* | Normalise internal absolute paths to `<REPO>` across 47 files and delete `scripts/_archive/` (43 one-off patch scripts that hard-coded the authors' workspace root). Recorded in `../PATH_SANITIZATION.md`. |
| 2 | `build.ps1` | Re-runnable form of step 1. |
| 3 | `compress.ps1` | Replace bulky result directories with single archives, then delete the loose copies. |
| 3b | *(one-off, not shipped)* | Rebuild the three evidence archives with their internal directory prefixes (`core/`, `e1/`, `selection/`) so extraction recreates the layout the frozen manifests refer to. Superseded the earlier prefix-less packs. |
| 4 | `repack.ps1` | Enforce the release-wide **no `.npz`** policy, and compress the bulky risk-audit JSON/stdout artefacts in `audit/risk_field/`. |
| 5 | `finalize.ps1` | Generate `PATH_SANITIZATION.md`, `RELEASE_ARCHIVES.md`, `RELEASE_SHA256SUMS.txt`, and `RELEASE_MANIFEST.json` — **in that order**, which matters (see below). |

## Why the generation order matters

`RELEASE_SHA256SUMS.txt` hashes the files that are loose **on disk in the published state**,
which necessarily includes the three `*_evidence.zip` archives. `RELEASE_MANIFEST.json`
instead records every *logical* file: loose files as stored, plus each archive member as it
appears **after extraction** (3 618 entries over 3 archives).

That makes `RELEASE_MANIFEST.json` the manifest `unpack_release.py` verifies against — a
sums file that lists the archives is useless once they have been expanded. It also means
`RELEASE_MANIFEST.json` cannot list its own hash: it is written last, after the sums file and
after the archives are final. Both files are therefore excluded from the sums list by
construction, and `verify_release.py` knows that.

## Packaging policy

* **Kept loose** — everything an executable entry point reads to run as documented, plus every
  human-readable report, table, figure and manuscript file.
* **Packaged** — three bulky *evidence trails* (1 549 files) whose value is completeness rather
  than readability: the R7 selection-stage trail and the two R11 forensic evidence sets.
* **Excluded** — all `.npz` transport/cost intermediates (regenerable; see
  `../EXTERNAL_DATA_MANIFEST.md`), the raw on-chain corpora and address-level labels, and the
  three credential-bearing config files found by the pre-publication scan
  (see `../SECURITY.md`).

No file in this release exceeds 18 MB, so **Git LFS is not used**.

## Reproducing the integrity manifests

```bash
python generate_sums.py     --root . --out RELEASE_SHA256SUMS.txt
python generate_manifest.py --root . --out RELEASE_MANIFEST.json
python verify_release.py
```

`generate_manifest.py` fails loudly if any two archives claim the same logical path.
