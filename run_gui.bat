@echo off
setlocal enabledelayedexpansion
title EduTerm AI Server

echo ==========================================================
echo Starting EduTerm AI Speech-to-Terminology Web GUI...
echo ==========================================================
cd /d "%~dp0experiments"

:start_server
python run_gui.py %*
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% NEQ 0 (
    echo.
    echo ==========================================================
    echo [CRASH DETECTED] Server process terminated with code %EXITCODE%.
    echo Auto-recovering: finding next open port and restarting in 2 seconds...
    echo (Press Ctrl+C to stop)
    echo ==========================================================
    timeout /t 2 /nobreak >nul
    goto start_server
)

pause
