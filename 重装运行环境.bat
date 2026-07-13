@echo off
setlocal EnableExtensions
chcp 65001 >nul
title QSflow - 重装环境
cd /d "%~dp0"

echo 此操作会删除本项目的 .venv 虚拟环境并重新安装依赖。
echo 不会删除图片、XML、模型或源代码。
echo.
choice /C YN /N /M "是否继续？[Y/N]: "
if errorlevel 2 exit /b 0

if exist "%~dp0.venv" rmdir /S /Q "%~dp0.venv"
call "%~dp0启动_QSflow.bat"
