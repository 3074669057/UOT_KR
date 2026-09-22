# repack.ps1 -- enforce "no .npz in the release tree" + compress bulky audit JSON
$ErrorActionPreference = 'Continue'
$REPO = 'D:\trae\tool\a\cross\_uot_kr_release\UOT_KR'
$TMP  = 'D:\trae\tool\a\cross\_uot_kr_release\_tmp'

Write-Host '=== A. repack R11_REMAINING6_CORE.zip without .npz ==='
$r11 = Join-Path $REPO 'out\handoff_r11_remaining6'
$coreZip = Join-Path $r11 'R11_REMAINING6_CORE.zip'
if (Test-Path $coreZip) {
    if (Test-Path $TMP) { Remove-Item $TMP -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $TMP | Out-Null
    Expand-Archive -Path $coreZip -DestinationPath $TMP -Force
    $npz = Get-ChildItem $TMP -Recurse -Force -File -Filter *.npz
    Write-Host ("  dropping {0} npz ({1} MB)" -f $npz.Count, [math]::Round((($npz | Measure-Object Length -Sum).Sum)/1MB,2))
    $npz | Remove-Item -Force
    $newZip = Join-Path $TMP 'R11_REMAINING6_CORE_nonpz.zip'
    Compress-Archive -Path (Join-Path $TMP '*') -DestinationPath $newZip -CompressionLevel Optimal -Force
    Copy-Item $newZip $coreZip -Force
    Write-Host ("  new R11_REMAINING6_CORE.zip = {0} MB" -f [math]::Round((Get-Item $coreZip).Length/1MB,2))
    Remove-Item $TMP -Recurse -Force
} else { Write-Host '  [SKIP] core zip missing' }

Write-Host ''
Write-Host '=== B. compress bulky audit JSON / stdout ==='
$audit = Join-Path $REPO 'audit\risk_field'
foreach ($n in @('discovery.json', 'nonconstant_scan.json', '06_stdout.txt', '05_stdout.txt')) {
    $p = Join-Path $audit $n
    if (Test-Path $p) {
        $b = (Get-Item $p).Length
        Compress-Archive -Path $p -DestinationPath "$p.zip" -CompressionLevel Optimal -Force
        $a = (Get-Item "$p.zip").Length
        Write-Host ("  {0,-24} {1,7} MB -> {2,6} MB" -f $n, [math]::Round($b/1MB,2), [math]::Round($a/1MB,2))
        Remove-Item $p -Force
    }
}

Write-Host ''
Write-Host '=== C. residual .npz check ==='
$left = Get-ChildItem $REPO -Recurse -Force -File -Filter *.npz
Write-Host ("  .npz files remaining in tree: {0}" -f $left.Count)
$left | Select-Object -First 10 | ForEach-Object { "    $($_.FullName)" }

Write-Host ''
$f = Get-ChildItem $REPO -Recurse -Force -File
Write-Host ("TREE NOW: {0} files, {1} MB" -f $f.Count, [math]::Round((($f | Measure-Object Length -Sum).Sum)/1MB, 2))
