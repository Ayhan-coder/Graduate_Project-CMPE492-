@echo off
setlocal enabledelayedexpansion
title EduTerm AI Defense Presentation

echo ==========================================================
echo Starting EduTerm AI Thesis Defense Presentation...
echo ==========================================================
cd /d "%~dp0presentation"

python serve_presentation.py %*
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% NEQ 0 (
    echo.
    echo ==========================================================
    echo [ERROR] Presentation server terminated with code %EXITCODE%.
    echo ==========================================================
    pause
)
