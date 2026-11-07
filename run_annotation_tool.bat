@echo off
chcp 65001 >nul
echo =====================================================================
echo   EduTerm AI - Altin Standart Etiketleme Studyosu
echo =====================================================================
echo.
echo Etiketleme arayuzu baslatiliyor ve tarayiciniz aciliyor...
python "%~dp0experiments\gui\annotator.py"
pause

