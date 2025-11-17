# GREMLIN Massive Language Pack Generator (PowerShell)
# Generates a 10,000 words/concept pack (~78 MB, 1.86M words)

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "GREMLIN Massive Language Pack Generator" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will generate a ~78 MB language pack with 1,860,000 words." -ForegroundColor Yellow
Write-Host "Generation will take 2-5 minutes." -ForegroundColor Yellow
Write-Host ""

$continue = Read-Host "Continue? (Y/N)"
if ($continue -ne "Y" -and $continue -ne "y") {
    Write-Host "Cancelled." -ForegroundColor Red
    exit
}

Write-Host ""
Write-Host "Starting generation..." -ForegroundColor Green
Write-Host ""

# Run the generator
python generate_language_pack.py --words 10000 --output language_packs/

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Done! Your language pack is ready." -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To use it with the admin console:" -ForegroundColor Yellow
Write-Host ""

# Get the newest pack file
$latestPack = Get-ChildItem language_packs\*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($latestPack) {
    Write-Host "    python demo\admin_console_tk.py --pack $($latestPack.FullName)" -ForegroundColor White
} else {
    Write-Host "    python demo\admin_console_tk.py --pack language_packs\language_pack_*.json" -ForegroundColor White
}

Write-Host ""
