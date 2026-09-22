# build.ps1 -- UOT_KR curated release staging
# Source (frozen):  D:\trae\tool\a\cross
# Stage  (repo tree): D:\trae\tool\a\cross\_uot_kr_release\UOT_KR
param(
    [switch]$SkipZip
)

$ErrorActionPreference = 'Continue'
$SRC   = 'D:\trae\tool\a\cross'
$ROOT  = 'D:\trae\tool\a\cross\_uot_kr_release'
$REPO  = Join-Path $ROOT 'UOT_KR'

# ---- exclusions applied to every copy -------------------------------------
$XD = @('__pycache__', '.pytest_cache', '.mypy_cache', '.ipynb_checkpoints', '.git', 'node_modules')
$XF = @('*.pyc', '*.pyo', '*.py.class', 'Thumbs.db', '.DS_Store', '*.tmp', '*.log.lock', '~$*')

function Copy-Tree {
    param([string]$RelPath, [string]$Dest, [string[]]$ExtraXD = @(), [string[]]$ExtraXF = @())
    $s = Join-Path $SRC $RelPath
    if (-not (Test-Path $s)) { Write-Host "  [SKIP-MISSING] $RelPath"; return }
    $xd = $XD + $ExtraXD
    $xf = $XF + $ExtraXF
    $args = @($s, $Dest, '/E', '/NFL', '/NDL', '/NJH', '/NJS', '/NP', '/R:1', '/W:1',
              '/XD') + $xd + @('/XF') + $xf
    & robocopy @args | Out-Null
    $code = $LASTEXITCODE
    if ($code -ge 8) { Write-Host "  [FAIL rc=$code] $RelPath" } else { Write-Host "  [ok rc=$code] $RelPath" }
    $global:LASTEXITCODE = 0
}

Write-Host '=== 1. code / config / docs (inline) ==='
foreach ($d in @('src', 'scripts', 'tests', 'config', 'schemas', 'docs')) {
    Copy-Tree -RelPath $d -Dest (Join-Path $REPO $d)
}

Write-Host '=== 2. result lines R5 / R6 (inline, no npz) ==='
foreach ($d in @('out\r5_posthoc_hparam_sensitivity_20260917',
                 'out\r6_posthoc_kernel_k_control_20260917')) {
    Copy-Tree -RelPath $d -Dest (Join-Path $REPO $d) -ExtraXF @('*.npz')
}

Write-Host '=== 3. R7 (inline, no npz) ==='
Copy-Tree -RelPath 'out\r7_confirmatory_kernel_ranking_20260917' `
          -Dest (Join-Path $REPO 'out\r7_confirmatory_kernel_ranking_20260917') `
          -ExtraXF @('*.npz')

Write-Host '=== 4. R11 handoff (inline, no npz) ==='
Copy-Tree -RelPath 'out\handoff_r11_remaining6' `
          -Dest (Join-Path $REPO 'out\handoff_r11_remaining6') `
          -ExtraXF @('*.npz')

Write-Host '=== 5. risk-field audit ==='
Copy-Tree -RelPath '_risk_audit_tmp' -Dest (Join-Path $REPO 'audit\risk_field')

Write-Host '=== 6. paper authoring package (R5, 409 files) ==='
Copy-Tree -RelPath 'ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE' -Dest (Join-Path $REPO 'paper\ZN_TIFS_R5_PAPER_AUTHORING_PACKAGE')

Write-Host '=== 7. submission-ready manuscript + hash list ==='
$pa = Join-Path $REPO 'paper'
New-Item -ItemType Directory -Force -Path $pa | Out-Null
$ma = Join-Path $SRC 'out\handoff_r11_remaining6\core\manuscript_authority'
if (Test-Path $ma) {
    Get-ChildItem $ma -File | Where-Object { $_.Name -notlike '*.npz' } | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $pa $_.Name) -Force
        Write-Host "  [ok] $($_.Name)"
    }
} else { Write-Host '  [WARN] manuscript_authority not found' }

Write-Host '=== done staging ==='
$f = Get-ChildItem $REPO -Recurse -Force -File
Write-Host ("STAGED: {0} files, {1} MB" -f $f.Count, [math]::Round((($f | Measure-Object Length -Sum).Sum)/1MB, 2))
