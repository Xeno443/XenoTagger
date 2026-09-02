@echo off
setlocal enabledelayedexpansion

REM Alternative to setup-portable.bat, for anyone who already has git and
REM Python installed systemwide and would rather use a normal venv than
REM this repo's portable environment (system\python) - run.cmd/cli.cmd
REM auto-detect and use whichever one exists. Doesn't touch system\ at
REM all; both setups can coexist side by side in the same checkout.
REM
REM The systemwide `python` on PATH may well be a different version than
REM the one the portable env pins (see setup-portable.bat) - webui/requirements.txt
REM has no version pins, so this is expected to just work across reasonably
REM recent Python versions, not something this script checks for.

set "ROOT=%~dp0"
set "VENV_DIR=%ROOT%.venv"

where python >nul 2>nul
if errorlevel 1 (
    echo No systemwide "python" found on PATH. Install Python first, or use
    echo setup-portable.bat instead if you'd rather not install anything systemwide.
    goto :fail
)

if exist "%VENV_DIR%\Scripts\python.exe" (
    echo .venv already exists, skipping creation.
) else (
    echo Creating venv at %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create venv.
        goto :fail
    )
)

echo.
echo Updating pip and its dependencies ...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo Failed to update pip.
    goto :fail
)

echo.
echo Installing Python dependencies for the tagger UI ...
"%VENV_DIR%\Scripts\python.exe" -m pip install -r "%ROOT%webui\requirements.txt"
if errorlevel 1 (
    echo Failed to install Python dependencies.
    goto :fail
)

echo.
echo Done. run.cmd / cli.cmd will use this .venv automatically. To finish
echo setup, install llama.cpp via the GUI's Settings Llama tab, or
echo cli.cmd --install-llama backend.
pause
goto :eof

:fail
echo.
echo Setup failed.
pause
exit /b 1
