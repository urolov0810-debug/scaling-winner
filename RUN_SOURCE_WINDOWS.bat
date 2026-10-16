@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py
    exit /b %errorlevel%
)
where py >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Install Python 3.11 or 3.12 first, or run BUILD_WINDOWS.bat.
    pause
    exit /b 1
)
py -3.12 main.py 2>nul
if errorlevel 1 py -3.11 main.py
if errorlevel 1 (
    echo [ERROR] Missing Python packages? Run BUILD_WINDOWS.bat, then use the EXE.
    pause
    exit /b 1
)
