param(
    [int]$Hours = 12,
    [int]$IntervalSeconds = 60,
    [string]$PythonExe = "D:\Data_Science\autoresearch\.venv\Scripts\python.exe",
    [switch]$SkipBootstrap
)

$ErrorActionPreference = "Continue"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$OutputEncoding = [System.Text.Encoding]::UTF8

$Workspace = Resolve-Path (Join-Path $PSScriptRoot "..")
$RepoRoot = Resolve-Path (Join-Path $Workspace "..\..\..")
$LogDir = Join-Path $Workspace "outputs\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir "rogii_controller_$Stamp.log"
$PidPath = Join-Path $LogDir "rogii_controller_$Stamp.pid"
$HeartbeatPath = Join-Path $LogDir "rogii_monitor_latest.json"
$StopPath = Join-Path $LogDir "STOP_ROGII_CONTROLLER"

$PythonRoot = Split-Path -Parent $PythonExe
$env:PATH = "$PythonRoot;$PythonRoot\DLLs;C:\Windows\System32;C:\Windows"
$env:PYTHONPATH = "$RepoRoot\src;$Workspace\src;$RepoRoot\.venv\Lib\site-packages"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:ROGII_RAW_DIR = "D:\Data_Science\data\rogii-wellbore-geology-prediction\raw"

# --- 資源ガバナの thread env を取り込む（OMP/BLAS + VRAM no-spill）。doc §8.2 / governed_launch ---
# 起動する python が lease の CPU 数（rogii の標準 lease）に揃った env を継承する。
try {
  $govEnv = & $PythonExe -m autoresearch.orchestration.thread_env --comp rogii-wellbore-geology-prediction
  if ($LASTEXITCODE -eq 0 -and $govEnv) { $govEnv | ForEach-Object { Invoke-Expression $_ } }
} catch { Write-Host "[governed] thread_env 取り込みをskip: $_" }

function Log([string]$msg) {
    $line = "[$(Get-Date -Format o)] $msg"
    [System.IO.File]::AppendAllText($LogPath, $line + "`r`n", (New-Object System.Text.UTF8Encoding($false)))
}

function Write-MonitorState([string]$state, [string]$detail) {
    $exp120 = Join-Path $Workspace "experiments\exp120_anchor_repro\result.json"
    $exp100 = Join-Path $Workspace "experiments\exp100_data_contract\result.json"
    $folds = Join-Path $Workspace "data\folds\folds_group_well_v002.csv"
    $processed = Join-Path $Workspace "data\processed\train_base_v001.parquet"
    $latestAction = Get-ChildItem -Path $LogDir -Filter "*.json" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne "rogii_monitor_latest.json" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    $cv = $null
    if (Test-Path $exp120) {
        try {
            $r = Get-Content -Raw -Encoding UTF8 $exp120 | ConvertFrom-Json
            $cv = $r.cv_rmse
        } catch {
            $cv = $null
        }
    }

    $payload = [ordered]@{
        ts = (Get-Date -Format o)
        state = $state
        detail = $detail
        controller_pid = $PID
        workspace = "$Workspace"
        stop_file = "$StopPath"
        checks = [ordered]@{
            raw_dir = (Test-Path $env:ROGII_RAW_DIR)
            processed_train = (Test-Path $processed)
            folds_v002 = (Test-Path $folds)
            data_contract_result = (Test-Path $exp100)
            anchor_result = (Test-Path $exp120)
            anchor_cv_rmse = $cv
        }
        latest_action_log = if ($latestAction) { "$($latestAction.FullName)" } else { $null }
    }
    $json = $payload | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText($HeartbeatPath, $json + "`r`n", (New-Object System.Text.UTF8Encoding($false)))
}

Log "ROGII controller started"
Log "repo=$RepoRoot"
Log "workspace=$Workspace"
Log "python=$PythonExe"
Log "hours=$Hours interval_seconds=$IntervalSeconds skip_bootstrap=$SkipBootstrap"
Log "stop_file=$StopPath"
$PID | Out-File -FilePath $PidPath -Encoding ascii
Write-MonitorState "starting" "controller initialized"

if (-not $SkipBootstrap) {
    Log "bootstrap start"
    Push-Location $RepoRoot
    try {
        $args = @(
            "competitions\rogii-wellbore-geology-prediction\workspace\scripts\rogii_autonomous_loop.py",
            "--mode", "bootstrap",
            "--python", $PythonExe
        )
        & $PythonExe @args *>> $LogPath
        $exit = $LASTEXITCODE
        Log "bootstrap exit_code=$exit"
        if ($exit -ne 0) {
            Write-MonitorState "bootstrap_failed" "bootstrap exit code $exit"
        } else {
            Write-MonitorState "bootstrap_completed" "bootstrap completed successfully"
        }
    } catch {
        Log "bootstrap exception=$($_.Exception.Message)"
        Write-MonitorState "bootstrap_exception" $_.Exception.Message
    } finally {
        Pop-Location
    }
}

$EndAt = (Get-Date).AddHours($Hours)
$Tick = 0
while ((Get-Date) -lt $EndAt) {
    if (Test-Path $StopPath) {
        Log "stop file detected; exiting"
        Write-MonitorState "stopping" "stop file detected"
        break
    }
    $Tick += 1
    Write-MonitorState "healthy" "monitor tick $Tick"
    Log "monitor tick=$Tick healthy"
    Start-Sleep -Seconds $IntervalSeconds
}

Write-MonitorState "finished" "controller finished"
Log "ROGII controller finished"
