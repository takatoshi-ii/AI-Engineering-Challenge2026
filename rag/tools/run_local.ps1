<#
Two-stage local run.

The two stages want opposite things from the GPU:

  OCR        many small independent images -> a small model, batched, several
             requests in flight at once
  Answering  one long growing conversation -> the largest model that fits, and
             only one 32k-context slot so the KV cache stays on the GPU

A 24 GB card cannot satisfy both at once, so the Ollama server is reconfigured
and restarted between them.

  .\run_local.ps1 vision  [-VisionModel qwen3-vl:8b]
  .\run_local.ps1 answer  [-AnswerModel qwen3.8:27b] [-Questions valid] [-Limit 0]
  .\run_local.ps1 all
#>
param(
    [Parameter(Position = 0)][ValidateSet('vision', 'answer', 'all')][string]$Stage = 'all',
    # qwen2.5vl:7b is the measured pick for this card: ~9s/image and stable.
    # qwen3-vl:8b spends its whole output budget on reasoning and returns empty
    # content; the 27B class does not fit alongside a KV cache and stalls.
    [string]$VisionModel = 'qwen2.5vl:7b',
    [string]$AnswerModel = 'qwen3.8:27b',
    [string]$Questions = 'valid',
    [int]$Limit = 0,
    [int]$VisionWorkers = 3
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
$env:OPENAI_BASE_URL = 'http://localhost:11434/v1'

function Restart-Ollama([int]$Parallel, [int]$Context) {
    Write-Host "== Ollama 再起動 (NUM_PARALLEL=$Parallel, CONTEXT=$Context) ==" -ForegroundColor Cyan
    [Environment]::SetEnvironmentVariable('OLLAMA_NUM_PARALLEL', "$Parallel", 'User')
    [Environment]::SetEnvironmentVariable('OLLAMA_CONTEXT_LENGTH', "$Context", 'User')
    [Environment]::SetEnvironmentVariable('OLLAMA_FLASH_ATTENTION', '1', 'User')
    [Environment]::SetEnvironmentVariable('OLLAMA_KV_CACHE_TYPE', 'q8_0', 'User')
    Get-Process ollama, 'ollama app' -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 4
    # the child inherits the values just written to the user environment
    $env:OLLAMA_NUM_PARALLEL = "$Parallel"
    $env:OLLAMA_CONTEXT_LENGTH = "$Context"
    $env:OLLAMA_FLASH_ATTENTION = '1'
    $env:OLLAMA_KV_CACHE_TYPE = 'q8_0'
    Start-Process -FilePath $Ollama -ArgumentList 'serve' -WindowStyle Hidden
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Seconds 2
        try {
            Invoke-WebRequest -Uri 'http://localhost:11434/api/version' -UseBasicParsing -TimeoutSec 5 | Out-Null
            Write-Host '   起動確認 OK'
            return
        } catch { }
    }
    throw 'Ollama が起動しませんでした'
}

function Show-Gpu {
    $u = (nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader)
    Write-Host "   GPU: $u" -ForegroundColor DarkGray
}

Push-Location $Root
try {
    if ($Stage -in @('vision', 'all')) {
        # small model, several images in flight; a short context is plenty
        # because each request is one image plus a fixed prompt
        Restart-Ollama -Parallel $VisionWorkers -Context 8192
        $env:RAG_MODEL = $VisionModel
        Write-Host "== 画像OCR ($VisionModel, workers=$VisionWorkers) ==" -ForegroundColor Green
        python -u rag/vision.py --workers $VisionWorkers
        Show-Gpu
    }

    if ($Stage -in @('answer', 'all')) {
        # one slot, full context: the agent transcript is what needs the room
        Restart-Ollama -Parallel 1 -Context 32768
        $env:RAG_MODEL = $AnswerModel
        Write-Host "== 回答生成 ($AnswerModel, $Questions) ==" -ForegroundColor Green
        $args = @('-u', 'rag/run.py', '--questions', $Questions, '--workers', '1')
        if ($Limit -gt 0) { $args += @('--limit', "$Limit") }
        python @args
        Show-Gpu
    }
} finally {
    Pop-Location
}
