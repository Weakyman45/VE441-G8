@echo off
setlocal

cd /d "%~dp0"

set "PORT=8000"
if not "%~1"=="" set "PORT=%~1"

set "DB_PATH=%CD%\catalog.db"
set "ENV_PATH=%CD%\backend\.env"

echo VoiceShop++ backend
echo Project: %CD%
echo Port:    %PORT%
echo.

if not exist "%DB_PATH%" (
    echo ERROR: catalog database not found:
    echo   %DB_PATH%
    echo.
    echo Place the enriched catalog.db in the project root, then run this script again.
    pause
    exit /b 1
)

if not exist "%ENV_PATH%" (
    echo WARNING: backend\.env not found.
    echo Realtime voice and image LLM parsing need DASHSCOPE_API_KEY.
    echo Create:
    echo   %ENV_PATH%
    echo with:
    echo   DASHSCOPE_API_KEY=your_key_here
    echo.
)

where py >nul 2>nul
if "%ERRORLEVEL%"=="0" (
    set "PY=py -3"
) else (
    where python >nul 2>nul
    if "%ERRORLEVEL%"=="0" (
        set "PY=python"
    ) else (
        echo ERROR: Python was not found. Install Python 3.10+ or add it to PATH.
        pause
        exit /b 1
    )
)

echo Starting backend...
echo URL: http://127.0.0.1:%PORT%
echo Health: http://127.0.0.1:%PORT%/health
echo.

%PY% backend\server.py --host 0.0.0.0 --port %PORT% --db "%DB_PATH%"

echo.
echo Backend stopped.
pause
