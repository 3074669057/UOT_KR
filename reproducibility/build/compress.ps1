# compress.ps1 -- replace bulky result directories with single archives
$ErrorActionPreference = 'Continue'
$REPO = 'D:\trae\tool\a\cross\_uot_kr_release\UOT_KR'

# rel path (relative to $REPO) -> zip file name placed next to the original dir's parent
$jobs = @(
    @{ Dir = 'out\r7_confirmatory_kernel_ranking_20260917\selection';                     Zip = 'selection.zip' },
    @{ Dir = 'out\r7_confirmatory_kernel_ranking_20260917\confirmatory';                  Zip = 'confirmatory.zip' },
    @{ Dir = 'out\r7_confirmatory_kernel_ranking_20260917\posthoc_s9_degree_stratification_20260918';  Zip = 'posthoc_s9_degree_stratification_20260918_full.zip' },
    @{ Dir = 'out\r7_confirmatory_kernel_ranking_20260917\posthoc_s10_unmatched_mass_localization_20260919'; Zip = 'posthoc_s10_unmatched_mass_localization_20260919_full.zip' },
    @{ Dir = 'out\handoff_r11_remaining6\core';                                           Zip = 'core_full.zip' },
    @{ Dir = 'out\handoff_r11_remaining6\e1';                                             Zip = 'e1_full.zip' }
)

foreach ($j in $jobs) {
    $src = Join-Path $REPO $j.Dir
    if (-not (Test-Path $src)) { Write-Host "[SKIP-MISSING] $($j.Dir)"; continue }
    $before = (Get-ChildItem $src -Recurse -Force -File | Measure-Object Length -Sum).Sum
    $zipPath = Join-Path (Split-Path $src -Parent) $j.Zip
    $t0 = Get-Date
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Compress-Archive -Path (Join-Path $src '*') -DestinationPath $zipPath -CompressionLevel Optimal -Force
    $after = (Get-Item $zipPath).Length
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    Write-Host ("[zip] {0,-70} {1,8} MB -> {2,7} MB  ({3}s)" -f $j.Dir, [math]::Round($before/1MB,2), [math]::Round($after/1MB,2), $secs)
    Remove-Item $src -Recurse -Force
}

Write-Host ''
$f = Get-ChildItem $REPO -Recurse -Force -File
Write-Host ("TREE NOW: {0} files, {1} MB" -f $f.Count, [math]::Round((($f | Measure-Object Length -Sum).Sum)/1MB, 2))
