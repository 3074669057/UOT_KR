# finalize.ps1 -- generate the release integrity + provenance documents
$ErrorActionPreference = 'Continue'
$ROOT = 'D:\trae\tool\a\cross\_uot_kr_release'
$REPO = Join-Path $ROOT 'UOT_KR'

# --------------------------------------------------------------------------- #
Write-Host '=== 1. PATH_SANITIZATION.md ==='
$san = Get-Content (Join-Path $ROOT 'PATH_SANITIZATION.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$md = New-Object System.Collections.Generic.List[string]
$md.Add('# PATH_SANITIZATION — internal absolute-path normalisation record')
$md.Add('')
$md.Add('The authors'' local workspace root was replaced by the placeholder `<REPO>` in the files')
$md.Add('listed below, so that the released artifacts carry no machine-specific paths.')
$md.Add('')
$md.Add('**No numeric value, threshold, metric or label was altered.** Only the literal path prefix')
$md.Add('`D:\trae\tool\a\cross` (and its forward-slash form) was rewritten. The before/after SHA256 of')
$md.Add('every touched file is recorded so the transformation is auditable.')
$md.Add('')
$md.Add(('Files rewritten: **{0}**' -f $san.Count))
$md.Add('')
$md.Add('| # | file | SHA256 before | SHA256 after |')
$md.Add('|---:|---|---|---|')
$i = 0
foreach ($e in ($san | Sort-Object path)) {
    $i++
    $md.Add(('| {0} | `{1}` | `{2}` | `{3}` |' -f $i, $e.path, $e.sha256_before, $e.sha256_after))
}
$md.Add('')
$md.Add('## Also removed from the release')
$md.Add('')
$md.Add('* `scripts/_archive/` (43 files) — one-off in-place patch scripts (e.g. `_fix_quotes.py`,')
$md.Add('  `_write_p8d.py`) that hard-coded the authors'' workspace root and edited source files.')
$md.Add('  They are not imported by, nor referenced from, any released driver, and they are not part')
$md.Add('  of the reproducible pipeline.')
$md.Add('')
$md.Add('## Reproduce')
$md.Add('')
$md.Add('```bash')
$md.Add('# after unpacking the release')
$md.Add('grep -rInE ''D:\\trae|C:\\Users\\'' . | head     # expected: no output')
$md.Add('```')
Set-Content -LiteralPath (Join-Path $REPO 'PATH_SANITIZATION.md') -Value $md -Encoding UTF8
Write-Host ("  wrote PATH_SANITIZATION.md ({0} entries)" -f $san.Count)

# --------------------------------------------------------------------------- #
Write-Host '=== 2. RELEASE_ARCHIVES.md ==='
$zips = Get-ChildItem $REPO -Recurse -Force -File -Filter *.zip |
        Where-Object { $_.FullName -notmatch '\\\.git\\' } | Sort-Object FullName
$amd = New-Object System.Collections.Generic.List[string]
$amd.Add('# RELEASE_ARCHIVES — packaged evidence trails')
$amd.Add('')
$amd.Add('Three bulky *evidence trails* ship as ZIP archives so the repository stays reviewable in')
$amd.Add('size. Nothing is lost: `python unpack_release.py` expands every archive **and** verifies')
$amd.Add('each extracted file against `RELEASE_SHA256SUMS.txt`.')
$amd.Add('')
$amd.Add('| archive | size | SHA256 | expands to |')
$amd.Add('|---|---:|---|---|')
foreach ($z in $zips) {
    $rel = $z.FullName.Substring($REPO.Length + 1) -replace '\\', '/'
    $h = (Get-FileHash -LiteralPath $z.FullName -Algorithm SHA256).Hash.ToLower()
    $amd.Add(('| `{0}` | {1} MB | `{2}` | in place |' -f $rel, [math]::Round($z.Length/1MB, 2), $h))
}
$amd.Add('')
$amd.Add('```bash')
$amd.Add('python unpack_release.py              # expand + verify (removes the archives)')
$amd.Add('python unpack_release.py --keep-zips  # expand + verify, keep the archives')
$amd.Add('python unpack_release.py --verify-only # verify the loose tree, expand nothing')
$amd.Add('```')
$amd.Add('')
$amd.Add('> Archive hashes are listed for reference only. ZIP stores member modification times, so')
$amd.Add('> re-building an archive from the same tree at a later time yields a *different* archive')
$amd.Add('> hash with *identical* member content. **`RELEASE_SHA256SUMS.txt` is the authoritative')
$amd.Add('> per-file integrity record**, not the archive hashes.')
Set-Content -LiteralPath (Join-Path $REPO 'RELEASE_ARCHIVES.md') -Value $amd -Encoding UTF8
Write-Host ("  wrote RELEASE_ARCHIVES.md ({0} archives)" -f $zips.Count)

# --------------------------------------------------------------------------- #
Write-Host '=== 2b. normalise generated docs to LF (before hashing) ==='
# Set-Content writes CRLF on Windows, but .gitattributes normalises .md files to LF in the
# repository.  If these two generated documents were hashed while still CRLF, the committed
# bytes would differ from the hashed bytes and verify_release.py would report them as
# mismatched on a fresh clone (exactly what happened: PATH_SANITIZATION.md and
# RELEASE_ARCHIVES.md).  Normalise BEFORE the hash manifests are written.
& python -c @"
import pathlib
root = pathlib.Path(r'$REPO')
changed = []
for rel in ('PATH_SANITIZATION.md', 'RELEASE_ARCHIVES.md'):
    p = root / rel
    if not p.is_file():
        continue
    b = p.read_bytes()
    if b'\r\n' in b or b'\r' in b:
        p.write_bytes(b.replace(b'\r\n', b'\n').replace(b'\r', b'\n'))
        changed.append(rel)
print('  normalised to LF:', changed or 'nothing to do')
"@
$global:LASTEXITCODE = 0

# --------------------------------------------------------------------------- #
Write-Host '=== 3. RELEASE_SHA256SUMS.txt (published on-disk state) ==='
# Generated by generate_sums.py (Python), NOT by a PowerShell hash loop: the release
# ships Chinese-named documents and PowerShell 5.1 would encode their paths in the
# console ANSI codepage, corrupting them to '?'.
#
# Order matters: this file must be written AFTER every content file exists, because it
# hashes them.  RELEASE_MANIFEST.json is therefore produced last and is deliberately not
# self-listed here (a manifest cannot contain its own hash).
& python (Join-Path $REPO 'generate_sums.py') --root $REPO --out RELEASE_SHA256SUMS.txt
if ($LASTEXITCODE -ne 0) { Write-Host '  [FAIL] generate_sums.py failed' }
$global:LASTEXITCODE = 0

Write-Host ''
Write-Host '=== 4. RELEASE_MANIFEST.json (canonical: loose files + archive members) ==='
# Written last so that its own recorded archive hashes are final.  It is the manifest
# unpack_release.py verifies against, because RELEASE_SHA256SUMS.txt necessarily includes
# the archives themselves and is therefore unusable after they are expanded.
& python (Join-Path $REPO 'generate_manifest.py') --root $REPO --out RELEASE_MANIFEST.json
if ($LASTEXITCODE -ne 0) { Write-Host '  [FAIL] generate_manifest.py failed' }
$global:LASTEXITCODE = 0

$all = Get-ChildItem $REPO -Recurse -Force -File | Where-Object { $_.FullName -notmatch '\\\.git\\' }
Write-Host ("`nTREE: {0} files, {1} MB" -f $all.Count, [math]::Round((($all | Measure-Object Length -Sum).Sum)/1MB, 2))
