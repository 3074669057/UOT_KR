# RELEASE_ARCHIVES 鈥?packaged evidence trails

Three bulky *evidence trails* ship as ZIP archives so the repository stays reviewable in
size. Nothing is lost: `python unpack_release.py` expands every archive **and** verifies
each extracted file against `RELEASE_SHA256SUMS.txt`.

| archive | size | SHA256 | expands to |
|---|---:|---|---|
| `audit/risk_field/05_stdout.txt.zip` | 0.05 MB | `e3571f155288155289cecb0c1bf762f0a1fa27b95d217ddcaa81cca16f9b4575` | in place |
| `audit/risk_field/06_stdout.txt.zip` | 0.04 MB | `ddd90d10358f5dba7ec010bb7ab14ce09b9b1b1f730cd020b5d31fac1643a1e2` | in place |
| `audit/risk_field/discovery.json.zip` | 0.54 MB | `efcba688b977e32db5f74998f6715c00d7b21e5f5ed37100e99072a66fbcb3f2` | in place |
| `audit/risk_field/nonconstant_scan.json.zip` | 0.17 MB | `137bd6f4cc0094afbc8f17c7b5f4dd1693cb0efe4758dc5036caf642f80abaf3` | in place |
| `out/handoff_r11_remaining6/R11_CORE_evidence.zip` | 17.74 MB | `360b4566c9f6b6ce25b50d8fbaf012edada3c5acf8ba45798fdff6cc983553d5` | in place |
| `out/handoff_r11_remaining6/R11_E1_evidence.zip` | 2.39 MB | `2484d2388cf121bcd97c2eda72f95e3738adbcfd1c63093384069ef4989f6817` | in place |
| `out/handoff_r11_remaining6/R11_REPORTS_AND_INDEX.zip` | 0.07 MB | `cbdff35f35d5472a4754bf6e2a5119f4b8bffb28164c5916916866b452a9f204` | in place |
| `out/r7_confirmatory_kernel_ranking_20260917/posthoc_s10_unmatched_mass_localization_20260919.zip` | 1.07 MB | `5d1eecf7beaad1db9860f3f156de4affa8f157a3bb7bca9dc80d34700e31e0df` | in place |
| `out/r7_confirmatory_kernel_ranking_20260917/posthoc_s9_degree_stratification_20260918.zip` | 1.09 MB | `f84055df495b460161127b394ff4de248c499d44a676fd0e69aa566dca67dc1d` | in place |
| `out/r7_confirmatory_kernel_ranking_20260917/posthoc_s9_degree_stratification_20260918/figures.zip` | 0.34 MB | `bb68b166e5cf6c19259d12f8b03e0babb63032fbeba574df61dc6bc9b6e4feef` | in place |
| `out/r7_confirmatory_kernel_ranking_20260917/selection_evidence.zip` | 4.22 MB | `97d02d8931b9a93c46afbcc5951df2807388253d294f6e94ffb284ee0883617a` | in place |

```bash
python unpack_release.py              # expand + verify (removes the archives)
python unpack_release.py --keep-zips  # expand + verify, keep the archives
python unpack_release.py --verify-only # verify the loose tree, expand nothing
```

> Archive hashes are listed for reference only. ZIP stores member modification times, so
> re-building an archive from the same tree at a later time yields a *different* archive
> hash with *identical* member content. **`RELEASE_SHA256SUMS.txt` is the authoritative
> per-file integrity record**, not the archive hashes.
