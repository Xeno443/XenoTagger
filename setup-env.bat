@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
set "SYSTEM_DIR=%ROOT%system"
set "TMP_DIR=%TEMP%\portable-env-setup"

set "WINPYTHON_VERSION=3.13.15.0"
set "WINPYTHON_EDITION=dot"
for /f "tokens=1,2 delims=." %%a in ("%WINPYTHON_VERSION%") do set "WINPYTHON_SERIES=%%a.%%b"
set "WINPYTHON_FILENAME=WinPython64-%WINPYTHON_VERSION%%WINPYTHON_EDITION%.exe"
set "WINPYTHON_URL=https://sourceforge.net/projects/winpython/files/WinPython_%WINPYTHON_SERIES%/%WINPYTHON_VERSION%/%WINPYTHON_FILENAME%/download"

set "GIT_TAG=v2.55.0.windows.5"
set "GIT_ASSET_VERSION=2.55.0.5"
set "GIT_URL=https://github.com/git-for-windows/git/releases/download/%GIT_TAG%/PortableGit-%GIT_ASSET_VERSION%-64-bit.7z.exe"

set "PYDIR=%SYSTEM_DIR%\python"

if not exist "%SYSTEM_DIR%" (
    echo Creating system folder ...
    md "%SYSTEM_DIR%"
)

if exist "%PYDIR%\python.exe" (
    echo Python already present, skipping.
) else (
    call :install_python
    if errorlevel 1 goto :fail
)

call :update_pip
if errorlevel 1 goto :fail

if exist "%SYSTEM_DIR%\git\bin\git.exe" (
    echo Git already present, skipping.
) else (
    call :install_git
    if errorlevel 1 goto :fail
)

echo.
echo Done.
pause
goto :eof

:install_python
echo.
echo Downloading WinPython %WINPYTHON_VERSION% (%WINPYTHON_EDITION%) ...
if not exist "%TMP_DIR%" md "%TMP_DIR%"
curl -L --fail "%WINPYTHON_URL%" -o "%TMP_DIR%\winpython.exe"
if errorlevel 1 (
    echo Failed to download WinPython.
    exit /b 1
)
echo Extracting WinPython ...
set "EXTRACT_DIR=%SYSTEM_DIR%\_winpython-extract"
if exist "%EXTRACT_DIR%" rd /s /q "%EXTRACT_DIR%"
"%TMP_DIR%\winpython.exe" -y -o"%EXTRACT_DIR%"

set "WPY_DIR="
for /d %%D in ("%EXTRACT_DIR%\WPy64-*") do set "WPY_DIR=%%D"
if not defined WPY_DIR (
    echo Could not locate extracted WinPython folder.
    exit /b 1
)

echo Moving Python runtime into place ...
move "%WPY_DIR%\python" "%PYDIR%" >nul
if errorlevel 1 (
    echo Failed to move Python runtime into place.
    exit /b 1
)

del /q "%TMP_DIR%\winpython.exe"
rd /s /q "%EXTRACT_DIR%"
exit /b 0

:update_pip
echo.
echo Updating pip and its dependencies ...
"%PYDIR%\python.exe" -m pip install --upgrade pip setuptools wheel packaging build pyproject_hooks colorama
if errorlevel 1 (
    echo Failed to update pip.
    exit /b 1
)
exit /b 0

:install_git
echo.
echo Downloading PortableGit %GIT_ASSET_VERSION% ...
if not exist "%TMP_DIR%" md "%TMP_DIR%"
curl -L --fail "%GIT_URL%" -o "%TMP_DIR%\PortableGit.exe"
if errorlevel 1 (
    echo Failed to download Git.
    exit /b 1
)
echo Extracting Git ...
if not exist "%SYSTEM_DIR%\git" md "%SYSTEM_DIR%\git"
"%TMP_DIR%\PortableGit.exe" -y -o"%SYSTEM_DIR%\git"
del /q "%TMP_DIR%\PortableGit.exe"
exit /b 0

:fail
echo.
echo Setup failed.
pause
exit /b 1
