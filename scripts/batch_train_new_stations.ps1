#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Batch train LightGBM v3 models for all new Thailand expansion stations.
.DESCRIPTION
    Sequentially trains 6h, 12h, 24h horizons for each new station with
    30 Optuna trials, saving to app/models/forecast_v3/.
#>

$ErrorActionPreference = "Stop"
$stations = @("SPB_01", "CEI_01", "LPT_01", "UDN_01", "NMA_01", "UBN_01", "JTI_01", "CBI_01", "HKT_01", "NST_01")
$horizons = "6,12,24"
$trials = 30
$scriptPath = "d:\Heat-wave-backend\scripts\train_forecast.py"
$logDir = "d:\Heat-wave-backend\logs\train_batch"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$results = @()
$startAll = Get-Date

foreach ($station in $stations) {
    $logFile = Join-Path $logDir "${station}.log"
    $stationStart = Get-Date
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Training $station ($horizons h, $trials trials)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    try {
        & python $scriptPath `
            --station $station `
            --horizons $horizons `
            --trials $trials `
            --model-version v3 `
            --backends lgbm `
            --device cpu `
            --no-skip 2>&1 | Tee-Object -FilePath $logFile

        $stationEnd = Get-Date
        $elapsed = ($stationEnd - $stationStart).TotalMinutes

        # Parse metrics from registry
        $h24Reg = "d:\Heat-wave-backend\app\models\forecast_v3\${station}\h24\registry.json"
        if (Test-Path $h24Reg) {
            $reg = Get-Content $h24Reg | ConvertFrom-Json
            $eval = $reg.evaluation
            $mae = $eval.regression.mae
            $skill = $eval.baselines.skill_score
            $status = $reg.status
            Write-Host "SUCCESS $station | MAE=${mae}°C Skill=${skill} Status=$status | ${elapsed:.1f} min" -ForegroundColor Green
            $results += [PSCustomObject]@{ Station=$station; Status="OK"; MAE=$mae; Skill=$skill; ModelStatus=$status; Minutes=[math]::Round($elapsed,1) }
        } else {
            Write-Host "SUCCESS $station (no registry) | ${elapsed:.1f} min" -ForegroundColor Green
            $results += [PSCustomObject]@{ Station=$station; Status="OK"; MAE=$null; Skill=$null; ModelStatus="unknown"; Minutes=[math]::Round($elapsed,1) }
        }
    } catch {
        $stationEnd = Get-Date
        $elapsed = ($stationEnd - $stationStart).TotalMinutes
        Write-Host "FAILED $station | ${elapsed:.1f} min | $_" -ForegroundColor Red
        $results += [PSCustomObject]@{ Station=$station; Status="FAIL"; MAE=$null; Skill=$null; ModelStatus="fail"; Minutes=[math]::Round($elapsed,1) }
    }
}

$endAll = Get-Date
$totalMin = ($endAll - $startAll).TotalMinutes

Write-Host "`n========================================" -ForegroundColor Yellow
Write-Host "BATCH TRAINING COMPLETE" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "Total time: ${totalMin:.1f} minutes"
$results | Format-Table -AutoSize

# Save summary
$summaryPath = Join-Path $logDir "summary.json"
$results | ConvertTo-Json -Depth 3 | Set-Content $summaryPath
Write-Host "Summary saved to $summaryPath"
