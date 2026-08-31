@echo off
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo .venv not found - run setup-venv.bat first.
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0webui\cli.py" %*
