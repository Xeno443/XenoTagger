@echo off
call environment.bat passive
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
set "REPO_URL=https://github.com/Xeno443/XenoTagger.git"

pushd "%ROOT%"

if exist "%ROOT%.git" goto :pull

echo No git checkout found here - adopting this folder as one ...
git init -b main
if errorlevel 1 goto :nogit
git remote add origin "%REPO_URL%"
git fetch origin
if errorlevel 1 goto :nogit
git reset --hard origin/main
if errorlevel 1 goto :nogit
git branch --set-upstream-to=origin/main main
goto :refresh_deps

:pull
set "current_branch="
for /f "delims=" %%C in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "current_branch=%%C"

if not defined current_branch (
    echo Current branch: [no branch information available]
) else if /i "!current_branch!"=="HEAD" (
    echo Current branch: [detached HEAD]
) else (
    echo Current branch: !current_branch!
)
echo Trying to update to latest version ...
git pull 2>NUL
if %ERRORLEVEL% == 0 goto :refresh_deps
echo git pull returned errorlevel %ERRORLEVEL%, initiating hard reset ...
git reset --hard
git pull

:refresh_deps
popd
if exist "%ROOT%webui\requirements.txt" (
    if exist "%ROOT%system\python\python.exe" (
        echo.
        echo Refreshing Python dependencies in the portable environment ...
        "%ROOT%system\python\python.exe" -m pip install -r "%ROOT%webui\requirements.txt"
    )
    if exist "%ROOT%.venv\Scripts\python.exe" (
        echo.
        echo Refreshing Python dependencies in .venv ...
        "%ROOT%.venv\Scripts\python.exe" -m pip install -r "%ROOT%webui\requirements.txt"
    )
)
goto :done

:nogit
popd
echo.
echo Could not set up a git checkout in this folder.
echo Make sure git is installed - run setup-portable.bat first, or install Git
echo for Windows yourself - and that you have an internet connection, then
echo try again.
pause
exit /b 1

:done
pause
