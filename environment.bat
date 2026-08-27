@echo off
set "DIR=%~dp0system"
ECHO Setting up portable environment using %DIR% ...
set "PATH=%DIR%\git\bin;%DIR%\python;%DIR%\python\Scripts;%PATH%"

if exist "%~dp0environment-local.bat" (
    call "%~dp0environment-local.bat"
)

if /I "%~1"=="passive" (
    REM we just set the env vars and return
    goto :EOF
)

ECHO.
ECHO ===================================================================
ECHO Use this command prompt to modify the embedded Python environment.
ECHO ===================================================================
PUSHD "%DIR%\python"
cmd.exe
POPD
EXIT
