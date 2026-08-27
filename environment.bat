@echo off
set "DIR=%~dp0system"
set "PATH=%DIR%\git\bin;%DIR%\python;%DIR%\python\Scripts;%PATH%"

set "GIT_BRANCH="
for /f "delims=" %%b in ('git -C "%~dp0." rev-parse --abbrev-ref HEAD 2^>nul') do set "GIT_BRANCH=%%b"
if not defined GIT_BRANCH set "GIT_BRANCH=unknown"

ECHO Setting up portable environment using %DIR% (branch: %GIT_BRANCH%) ...
title portable-env [%GIT_BRANCH%]

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
