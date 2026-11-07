$Host.UI.RawUI.WindowTitle = "EduTerm AI - Altın Standart Etiketleme Stüdyosu"
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  EduTerm AI - Altın Standart Etiketleme Stüdyosu" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Etiketleme arayüzü başlatılıyor ve tarayıcınız açılıyor..." -ForegroundColor Yellow
python "$PSScriptRoot\experiments\gui\annotator.py"

