# Windows version of run_all.sh. Run from the repo root with the env activated:
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1
# Output goes to the console and to logs\. A failed part does not stop the rest.
Set-Location $PSScriptRoot
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force -Path logs, results, checkpoints | Out-Null

$parts = @("part3_yolov8", "part2_faster_rcnn", "part4_deeplabv3", "part5_mask_rcnn", "part6_analysis")
foreach ($part in $parts) {
    Write-Host "=== START $part $(Get-Date) ==="
    python -u "$part.py" 2>&1 | Tee-Object -FilePath "logs\$part.log"
    Write-Host "=== END $part exit $LASTEXITCODE $(Get-Date) ==="
}
Write-Host "=== ALL DONE $(Get-Date) ==="
