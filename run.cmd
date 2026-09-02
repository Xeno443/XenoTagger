@echo off
if exist "%~dp0system\python\python.exe" goto :portable
if exist "%~dp0.venv\Scripts\python.exe" goto :venv

echo No environment found. Run setup-portable.bat or setup-venv.bat first.
pause
exit /b 1

:portable
call "%~dp0environment.bat" passive
"%~dp0system\python\python.exe" "%~dp0webui\app.py"
pause
exit /b 0

:venv
"%~dp0.venv\Scripts\python.exe" "%~dp0webui\app.py"
pause
exit /b 0
