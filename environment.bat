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

set "WRAPPER_FIX=%DIR%\fix-wrappers.py"

if not exist "%DIR%\python\python.exe" goto :after_wrapper_fix
if not exist "%WRAPPER_FIX%" goto :after_wrapper_fix

"%DIR%\python\python.exe" "%WRAPPER_FIX%" "%DIR%\python\Scripts" >"%TEMP%\xenotagger-wrapper-fix.log" 2>&1
findstr /C:"fixed (was broken)" "%TEMP%\xenotagger-wrapper-fix.log" >nul
if not errorlevel 1 (
    ECHO Some script wrappers had a stale interpreter path - fixed:
    findstr /C:"fixed (was broken)" "%TEMP%\xenotagger-wrapper-fix.log"
)
del "%TEMP%\xenotagger-wrapper-fix.log" >nul 2>&1

:after_wrapper_fix

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
