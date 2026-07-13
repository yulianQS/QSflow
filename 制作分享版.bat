@echo off
setlocal EnableExtensions
chcp 65001 >nul
title QSflow - 制作分享版
cd /d "%~dp0"

set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined PYTHON_EXE where python >nul 2>nul && set "PYTHON_EXE=python"
if not defined PYTHON_EXE where py >nul 2>nul && set "PYTHON_EXE=py -3"
if not defined PYTHON_EXE (
    echo 未找到 Python，无法制作分享版。
    pause
    exit /b 1
)

%PYTHON_EXE% "%~dp0tools\build_portable.py"
if errorlevel 1 (
    echo 制作失败，请检查上方错误。
    pause
    exit /b 1
)

echo.
echo 分享版已经生成到：dist\QSflow-portable
echo 将整个 QSflow-portable 文件夹压缩后发送给他人即可。
pause
exit /b 0

