@echo off
setlocal
cd /d "%~dp0"
if not exist "dist\CLPT_FSDT_RK\CLPT_FSDT_RK.exe" (
  echo [ERROR] First run BUILD_WINDOWS.bat to produce the .exe.
  pause
  exit /b 1
)
set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (
  echo [ERROR] Inno Setup 6 compiler not found. Install it from jrsoftware.org.
  pause
  exit /b 1
)
"%ISCC%" installer.iss
if errorlevel 1 (
  echo [ERROR] Installer build failed.
  pause
  exit /b 1
)
echo [OK] Setup.exe is in the installer_output folder.
pause
