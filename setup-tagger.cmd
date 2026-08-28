@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
set "LLAMA_DIR=%ROOT%llama-cuda"
set "TMP_DIR=%TEMP%\portable-env-tagger-setup"
set "PYEXE=%ROOT%system\python\python.exe"

rem Bump LLAMA_BUILD to pick up a newer llama.cpp release:
rem https://github.com/ggml-org/llama.cpp/releases
rem TODO: hardcoded to one CUDA version for now; revisit to let the user
rem pick (older cards / drivers may need 12.4 instead of 13.x).
set "LLAMA_BUILD=b10656"
set "LLAMA_CUDA_VERSION=13.3"
set "LLAMA_BIN_FILENAME=llama-%LLAMA_BUILD%-bin-win-cuda-%LLAMA_CUDA_VERSION%-x64.zip"
set "LLAMA_CUDART_FILENAME=cudart-llama-bin-win-cuda-%LLAMA_CUDA_VERSION%-x64.zip"
set "LLAMA_BIN_URL=https://github.com/ggml-org/llama.cpp/releases/download/%LLAMA_BUILD%/%LLAMA_BIN_FILENAME%"
set "LLAMA_CUDART_URL=https://github.com/ggml-org/llama.cpp/releases/download/%LLAMA_BUILD%/%LLAMA_CUDART_FILENAME%"

if exist "%LLAMA_DIR%\llama-server.exe" (
    echo llama.cpp CUDA build already present in "%LLAMA_DIR%", skipping.
) else (
    call :install_llama
    if errorlevel 1 goto :fail
)

call :install_python_deps
if errorlevel 1 goto :fail

echo.
echo Done.
pause
goto :eof

:install_llama
if not exist "%TMP_DIR%" md "%TMP_DIR%"
if not exist "%LLAMA_DIR%" md "%LLAMA_DIR%"

echo.
echo Downloading llama.cpp %LLAMA_BUILD% (CUDA %LLAMA_CUDA_VERSION%, win-x64) ...
curl -L --fail "%LLAMA_BIN_URL%" -o "%TMP_DIR%\%LLAMA_BIN_FILENAME%"
if errorlevel 1 (
    echo Failed to download llama.cpp CUDA build.
    exit /b 1
)

echo.
echo Downloading matching CUDA runtime (cudart/cublas) ...
curl -L --fail "%LLAMA_CUDART_URL%" -o "%TMP_DIR%\%LLAMA_CUDART_FILENAME%"
if errorlevel 1 (
    echo Failed to download CUDA runtime package.
    exit /b 1
)

echo.
echo Extracting llama.cpp into "%LLAMA_DIR%" ...
tar -xf "%TMP_DIR%\%LLAMA_BIN_FILENAME%" -C "%LLAMA_DIR%"
if errorlevel 1 (
    echo Failed to extract llama.cpp build.
    exit /b 1
)

echo Extracting CUDA runtime into "%LLAMA_DIR%" ...
tar -xf "%TMP_DIR%\%LLAMA_CUDART_FILENAME%" -C "%LLAMA_DIR%"
if errorlevel 1 (
    echo Failed to extract CUDA runtime package.
    exit /b 1
)

del /q "%TMP_DIR%\%LLAMA_BIN_FILENAME%"
del /q "%TMP_DIR%\%LLAMA_CUDART_FILENAME%"
exit /b 0

:install_python_deps
echo.
echo Installing Python dependencies for the tagger UI ...
"%PYEXE%" -m pip install -r "%ROOT%webui\requirements.txt"
if errorlevel 1 (
    echo Failed to install Python dependencies.
    exit /b 1
)
exit /b 0

:fail
echo.
echo Setup failed.
pause
exit /b 1
