#!/usr/bin/env pwsh
# Start the OpenAI-compatible exllamav3 server (tools/serve_openai.py).
# Native Windows port of start.sh. Configuration lives in .env.
# First run needs cl.exe + nvcc on PATH: launch from a VS Developer
# prompt (vcvars64.bat, x64) so the CUDA extension precompile can build.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Test-Path tools/serve_openai.py)) {
    Write-Error 'tools/serve_openai.py not found. Run from the engine repo or deployment kit.'
    exit 1
}
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host 'Created .env from .env.example. Edit it (HF_TOKEN, MODEL_DIR) and rerun.'
    exit 1
}

# Load .env (KEY=VALUE, # comments). Same file format as start.sh.
$cfg = @{}
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -notmatch '=') {
        return
    }
    $k, $v = $_.Split('=', 2)
    $cfg[$k.Trim()] = $v.Trim()
}
if ($cfg['HF_TOKEN']) {
    $env:HF_TOKEN = $cfg['HF_TOKEN']
}

$VenvPython = Join-Path $PWD '.venv/Scripts/python.exe'
function Test-EngineImport {
    & $VenvPython -c 'import torch, exllamav3, aiohttp, huggingface_hub, transformers' 2>$null
    return $LASTEXITCODE -eq 0
}
if (-not (Test-Path $VenvPython) -or -not (Test-EngineImport)) {
    Write-Host 'First-run setup (later runs skip to the model):'
    python -m venv .venv
    & $VenvPython -m pip install --quiet --upgrade pip setuptools wheel typing_extensions packaging
    $torchIndex = $cfg['TORCH_INDEX_URL']
    if (-not $torchIndex) {
        $torchIndex = 'https://download.pytorch.org/whl/cu130'
    }
    & $VenvPython -m pip install torch --extra-index-url $torchIndex
    & $VenvPython -m pip install triton-windows
    # Engine deps minus flash-linear-attention (no Windows wheel; unused by
    # the Qwen3.8 qwen3/dflash2/MTP path). --no-deps on the engine install
    # below keeps pip from resolving it. transformers renders the HF chat
    # template at request time.
    & $VenvPython -m pip install tokenizers numpy rich typing_extensions safetensors ninja pillow pyyaml marisa_trie pydantic 'llguidance>=1.7.0' transformers
    # The precompile needs the x64 MSVC tools + matching env. Without cl.exe
    # the extension cannot build, so stop instead of serving a broken venv.
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        Write-Error 'cl.exe not on PATH. Rerun from a VS Developer prompt (vcvars64.bat, x64) for the CUDA precompile.'
        exit 1
    }
    if (-not $env:CUDA_HOME) {
        $env:CUDA_HOME = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
    }
    if (-not $env:MAX_JOBS) {
        $env:MAX_JOBS = '4'
    }
    $env:DISTUTILS_USE_SDK = '1'
    & $VenvPython -m pip install --no-deps --no-build-isolation .
    & $VenvPython -m pip install --quiet aiohttp huggingface_hub
    Write-Host 'Setup complete.'
}

$PYTHON = $VenvPython
$env:PATH = (Join-Path $PWD '.venv/Scripts') + ';' + $env:PATH

$MODEL_DIR = $cfg['MODEL_DIR']; if (-not $MODEL_DIR) {
    Write-Error 'MODEL_DIR must be set in .env'; exit 1
}
$PORT = $cfg['PORT']; if (-not $PORT) {
    $PORT = '8888'
}
$HOST_ = $cfg['HOST']; if (-not $HOST_) {
    $HOST_ = '127.0.0.1'
}
$CONTEXT_SIZE = [int]($cfg['CONTEXT_SIZE']); if (-not $CONTEXT_SIZE) {
    $CONTEXT_SIZE = 65536
}
$GPU_MEM_GB = $cfg['GPU_MEM_GB']
if (-not $GPU_MEM_GB) {
    $vram = (nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
    if ([int]$vram -gt 0) {
        $GPU_MEM_GB = [string]([int]$vram / 1024 - 2)
    } else {
        $GPU_MEM_GB = '22'
    }
    Write-Host "GPU_MEM_GB not set. Budget: $GPU_MEM_GB GB (override in .env)"
}
$CACHE_QUANT = $cfg['CACHE_QUANT']; if (-not $CACHE_QUANT) {
    $CACHE_QUANT = 'none'
}
$CPU_CACHE_GB = $cfg['CPU_CACHE_GB']; if (-not $CPU_CACHE_GB) {
    $CPU_CACHE_GB = '0'
}

$DRAFT = $cfg['DRAFT']
if (-not $DRAFT) {
    if ($cfg['DRAFT_DIR'] -eq 'none') {
        $DRAFT = 'none'
    } elseif ($cfg['DRAFT_DIR']) {
        $DRAFT = 'dflash2'
    } else {
        $DRAFT = 'mtp'
    }
}
$DRAFT = $DRAFT.ToLower()

$HF_TARGET_REPO = $cfg['HF_TARGET_REPO']; if (-not $HF_TARGET_REPO) {
    $HF_TARGET_REPO = 'Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw'
}
$HF_DRAFT_REPO = $cfg['HF_DRAFT_REPO']; if (-not $HF_DRAFT_REPO) {
    $HF_DRAFT_REPO = 'Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw'
}

function Get-Model($repo, $dir, $label) {
    if ((Test-Path (Join-Path $dir 'config.json')) -and (Get-ChildItem (Join-Path $dir '*.safetensors') -ErrorAction SilentlyContinue)) {
        Write-Host "$label`: $dir present. Skip download."
        return
    }
    Write-Host "$label`: download $repo to $dir ..."
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    & $PYTHON -c 'from huggingface_hub import snapshot_download; import sys; print(snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2]))' $repo $dir
}

Get-Model -Repo $HF_TARGET_REPO -Dir $MODEL_DIR -Label 'target model'
switch ($DRAFT) {
    'mtp' {
    }
    'dflash2' {
        $draftDir = $cfg['DRAFT_DIR']
        if (-not $draftDir -or $draftDir -eq 'none') {
            $draftDir = 'models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw'
        }
        Get-Model -Repo $HF_DRAFT_REPO -Dir $draftDir -Label 'DFlash2 draft'
        $cfg['DRAFT_DIR'] = $draftDir
    }
    'none' {
    }
    default {
        Write-Error "DRAFT must be mtp, dflash2, or none (got: $DRAFT)"; exit 1
    }
}

if ($CONTEXT_SIZE -gt 262144 -and (Test-Path (Join-Path $MODEL_DIR 'config.yarn-1m.json'))) {
    $conf = Get-Content (Join-Path $MODEL_DIR 'config.json') -Raw
    if ($conf -notmatch 'rope_scaling') {
        Copy-Item (Join-Path $MODEL_DIR 'config.yarn-1m.json') (Join-Path $MODEL_DIR 'config.json')
        Write-Host 'CONTEXT_SIZE > 262k: switched config.json to YaRN 1M variant.'
    }
}

$cmd = @($PYTHON, '-u', 'tools/serve_openai.py', '--model', $MODEL_DIR,
    '--host', $HOST_, '--port', $PORT, '--cache_size', "$CONTEXT_SIZE", '--grid_size', "$GPU_MEM_GB")
if ($CACHE_QUANT -ne 'none') {
    $cmd += @('--cache_quant', $CACHE_QUANT)
}
switch ($DRAFT) {
    'mtp' {
        $cmd += @('--draft_model', 'mtp')
    }
    'dflash2' {
        $cmd += @('--draft_model', $cfg['DRAFT_DIR'])
    }
    'none' {
        $cmd += @('--draft_model', 'none')
    }
}
if ($CPU_CACHE_GB -ne '0') {
    $cmd += @('--cpu_cache_size', $CPU_CACHE_GB)
}

Write-Host "Starting: $($cmd -join ' ')"
& $cmd[0] $cmd[1..($cmd.Count - 1)]

