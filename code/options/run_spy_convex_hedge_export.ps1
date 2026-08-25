$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "SPY Convex Hedge Data Export"
Write-Host "Your MarketData token will not be written to disk."
Write-Host ""

$secure = Read-Host "Paste MarketData token" -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)

try {
    $env:MARKETDATA_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)

    py .\marketdata_spy_convex_hedge_exporter.py `
        --start 2010-01-01 `
        --end 2026-08-24 `
        --roll-rule first `
        --dtes 60,90 `
        --workers 8 `
        --out MarketData_SPY_Convex_Hedge
}
finally {
    if ($ptr -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    }
    Remove-Item Env:MARKETDATA_TOKEN -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Finished. Output folder: MarketData_SPY_Convex_Hedge"
Read-Host "Press Enter to close"
