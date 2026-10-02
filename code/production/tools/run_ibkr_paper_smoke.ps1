param(
  [Parameter(Mandatory=$true)]
  [string]$PaperAccountId,

  [ValidateSet(7497,4002)]
  [int]$Port = 7497,

  [int]$ClientId = 97,

  [switch]$ConnectionOnly
)

$ErrorActionPreference = "Stop"

if (-not $PaperAccountId.ToUpperInvariant().StartsWith("DU")) {
  throw "PaperAccountId must be DU-prefixed. Refusing to continue."
}

$Root = Split-Path -Parent $PSScriptRoot
$ConfigTemplate = Join-Path $Root "config\ibkr_paper_smoke.example.json"
$StateDir = Join-Path $Root "state"
$LocalConfig = Join-Path $StateDir "ibkr_paper_smoke.local.json"

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

$cfg = Get-Content $ConfigTemplate -Raw | ConvertFrom-Json
$cfg.account.account_id = $PaperAccountId
$cfg.broker.paper_port = $Port
$cfg.broker.client_id = $ClientId
$cfg.state.sqlite_path = (Join-Path $StateDir "ibkr_paper_smoke.sqlite3")
$cfg | ConvertTo-Json -Depth 20 | Set-Content -Encoding UTF8 $LocalConfig

Write-Host "Created untracked local PAPER config at $LocalConfig"
Write-Host "Running IBKR connection probe on 127.0.0.1:$Port clientId=$ClientId"

py (Join-Path $Root "tools\ibkr_connection_probe.py") --host 127.0.0.1 --port $Port --client-id $ClientId

if ($LASTEXITCODE -ne 0) {
  throw "IBKR paper connection probe failed."
}

if ($ConnectionOnly) {
  Write-Host "Connection-only check passed. No order submitted."
  exit 0
}

Write-Host "Submitting bounded PAPER smoke order: BUY 1 SPY MKT DAY."
Write-Host "If filled, the harness will submit the opposite quantity to restore the starting SPY position."

py (Join-Path $Root "tools\ibkr_paper_order_smoke.py") --config $LocalConfig --ack PAPER-ONLY-1SHARE-SPY

if ($LASTEXITCODE -ne 0) {
  throw "IBKR paper order smoke failed. Review the JSON audit in state\ before taking any further action."
}

Write-Host "IBKR PAPER smoke completed successfully."
