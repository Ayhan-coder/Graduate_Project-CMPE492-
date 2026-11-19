@echo off
set SDK=C:\Users\Slayer\AppData\Local\Android\Sdk
set ADB=%SDK%\platform-tools\adb.exe
set EMU=%SDK%\emulator\emulator.exe
set APK=%~dp0android\app\build\outputs\apk\debug\app-debug.apk

echo [1/3] Android Emülatör (knb_phone) başlatılıyor...
start "" "%EMU%" -avd knb_phone

echo [2/3] Emülatörün açılması bekleniyor...
"%ADB%" wait-for-device
:waitboot
for /f "delims=" %%a in ('"%ADB%" shell getprop sys.boot_completed 2^>nul') do set BOOT=%%a
if not "%BOOT%"=="1" (
    timeout /t 2 /nobreak >nul
    goto waitboot
)

echo [3/3] Uygulama kuruluyor ve açılıyor...
"%ADB%" install -r "%APK%"
"%ADB%" shell input keyevent 82
"%ADB%" shell am start -n edu.boun.cmpe492.edutermeval/.MainActivity
echo Hazir! Uygulama emulatorde acildi.
pause
