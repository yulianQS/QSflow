@echo off
rem 旧文件名兼容入口：统一转到 QSflow 主启动器，避免两套环境逻辑不一致。
call "%~dp0启动_QSflow.bat" %*
exit /b %ERRORLEVEL%
