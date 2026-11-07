@echo off
chcp 65001 >nul
echo =====================================================================
echo   EduTerm AI - Son 2 Hafta Gelistirmeleri Toplanti Sunumu
echo =====================================================================
echo.
echo Sunum varsayilan tarayicinizda aciliyor...
start "" "%~dp0presentation\toplanti_2hafta_sunum.html"
echo.
echo Klavye Kisayollari:
echo   [Sag Ok / Bosluk] : Sonraki Slayt
echo   [Sol Ok]          : Onceki Slayt
echo   [N Tusu]          : Konusmaci Notlarini Ac / Kapat
echo   [F Tusu]          : Tam Ekran (Fullscreen)
echo.
echo Basarili sunumlar dileriz!
timeout /t 5 >nul

