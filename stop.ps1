#!/usr/bin/env pwsh
# stop.ps1 - stop the server started by start.ps1 (graceful, then forced).
# Windows counterpart to stop.sh. Reads PORT from .env (default 8888).
# Safe to run when nothing is up. Refuses to kill a non-server listener.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$PORT = $env:PORT
if (-not $PORT -and (Test-Path .env)) {
    $line = Get-Content .env | Where-Object { $_ -match '^\s*PORT\s*=' } | Select-Object -First 1
    if ($line) {
        $PORT = $line.Split('=', 2)[1].Trim()
    }
}
if (-not $PORT) {
    $PORT = '8888'
}

$conn = Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $conn) {
    Write-Host "No server listening on port $PORT."
    exit 0
}

$proc = Get-CimInstance Win32_Process -Filter "ProcessId = $($conn.OwningProcess)" -ErrorAction SilentlyContinue
if (-not $proc -or $proc.CommandLine -notmatch 'serve_openai') {
    Write-Host "Port $PORT is held by another process (pid $($conn.OwningProcess))."
    Write-Host 'Refusing to kill it.'
    exit 1
}

Write-Host "Stopping server (pid $($conn.OwningProcess)) on port $PORT..."
taskkill /PID $conn.OwningProcess | Out-Null
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Milliseconds 500
    $still = Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue
    if (-not $still) {
        Write-Host 'Stopped.'
        exit 0
    }
}

Write-Host 'Still running after 10s, forcing...'
taskkill /F /PID $conn.OwningProcess | Out-Null
Start-Sleep 1
if (Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue) {
    Write-Host "ERROR: port $PORT still open."
    exit 1
}
Write-Host 'Stopped.'

