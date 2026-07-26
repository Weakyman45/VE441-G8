@echo off
rem VoiceShop++ backend launcher (native Windows, no WSL).
rem Secrets are read from backend\.env by server.py automatically.
setlocal
cd /d "%~dp0"

rem --- locate a Python 3 interpreter ---
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY if exist "D:\python.exe" set "PY=D:\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY (
    echo [ERROR] No Python 3 found. Install Python 3.12 first.
    exit /b 1
)

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"

echo Starting VoiceShop++ backend with: %PY%
echo Port: %PORT%   ^(emulator reaches it at http://10.0.2.2:%PORT%^)
echo Press Ctrl+C to stop.
echo.
%PY% server.py --host 0.0.0.0 --port %PORT%
