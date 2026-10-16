@echo off
setlocal
cd /d "%~dp0"

echo === CLPT + FSDT + RK Windows app builder ===
echo Building on Windows 10/11 x64 with Python 3.11 or 3.12.

where py >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Windows Python launcher "py" not found. Install Python 3.11 or 3.12 x64.
  pause
  exit /b 1
)

py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    py -3.12 -m venv .venv
) else (
    py -3.11 -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Install Python 3.12 or 3.11 64-bit.
        pause
        exit /b 1
    )
    py -3.11 -m venv .venv
)
if errorlevel 1 goto fail

".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements-build.txt
if errorlevel 1 goto fail

echo.
echo === Packaging desktop EXE (no console) ===
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --windowed --onedir --name CLPT_FSDT_RK --collect-all scipy --collect-all matplotlib main.py
if errorlevel 1 goto fail

echo.
echo [OK] Portable EXE: dist\CLPT_FSDT_RK\CLPT_FSDT_RK.exe
echo Copy the WHOLE dist\CLPT_FSDT_RK directory to another x64 Windows PC.
echo To make Setup.exe, install Inno Setup 6 and run BUILD_INSTALLER.bat.
pause
exit /b 0

:fail
echo [ERROR] Build failed; review the error messages above.
pause
exit /b 1
