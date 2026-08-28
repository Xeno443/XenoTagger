@echo off
call "%~dp0environment.bat" passive
"%~dp0system\python\python.exe" "%~dp0webui\cli.py" %*
