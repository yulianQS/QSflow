@echo off
setlocal EnableExtensions
chcp 65001 >nul
title QSflow
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

set "VENV_DIR=%~dp0.venv"
set "PYTHON_EXE="
set "PYTHON_ARGS="

echo Searching for an existing compatible Python environment...

if exist "%VENV_DIR%\Scripts\python.exe" (
    call :check_python "%VENV_DIR%\Scripts\python.exe"
    if not errorlevel 1 (
        set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
        echo Using project environment: .venv
        goto :launch
    )
)

if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
    call :check_python "%VIRTUAL_ENV%\Scripts\python.exe"
    if not errorlevel 1 (
        set "PYTHON_EXE=%VIRTUAL_ENV%\Scripts\python.exe"
        echo Using active virtual environment: %VIRTUAL_ENV%
        goto :launch
    )
)

if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
    call :check_python "%CONDA_PREFIX%\python.exe"
    if not errorlevel 1 (
        set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
        echo Using active Conda environment: %CONDA_PREFIX%
        goto :launch
    )
)

where python >nul 2>nul
if not errorlevel 1 (
    call :check_python python
    if not errorlevel 1 (
        set "PYTHON_EXE=python"
        echo Using Python from PATH.
        goto :launch
    )
)

where py >nul 2>nul
if not errorlevel 1 (
    call :check_python_launcher
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "PYTHON_ARGS=-3"
        echo Using Windows Python Launcher.
        goto :launch
    )
)

echo No existing environment contains all required packages.
echo Creating an isolated .venv in the project directory.

set "BASE_PYTHON="
where python >nul 2>nul && set "BASE_PYTHON=python"
if not defined BASE_PYTHON where py >nul 2>nul && set "BASE_PYTHON=py -3"
if not defined BASE_PYTHON (
    echo ERROR: Python was not found. Install Python 3.10, 3.11, or 3.12.
    goto :failed
)

%BASE_PYTHON% -c "import sys; exit(0 if sys.version_info[:2] in [(3,10),(3,11),(3,12)] else 1)" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python 3.10, 3.11, or 3.12 is required.
    %BASE_PYTHON% --version
    goto :failed
)

if exist "%VENV_DIR%" (
    echo ERROR: An incomplete .venv exists. Run the environment repair BAT first.
    goto :failed
)

echo [1/2] Creating virtual environment...
%BASE_PYTHON% -m venv "%VENV_DIR%"
if errorlevel 1 goto :failed

echo [2/2] Installing dependencies. The first run may take several minutes...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
"%VENV_DIR%\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :failed

set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
echo Environment installation completed.

:launch
echo Starting QSflow...
"%PYTHON_EXE%" %PYTHON_ARGS% "%~dp0app.py" %*
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" (
    echo ERROR: Application exited with code %APP_EXIT%.
    goto :failed
)
endlocal
exit /b 0

:check_python
"%~1" -c "import sys; import PySide6, PIL, ultralytics, openai, yaml, ai_edge_litert; exit(0 if sys.version_info[:2] in [(3,10),(3,11),(3,12)] else 1)" >nul 2>nul
exit /b %ERRORLEVEL%

:check_python_launcher
py -3 -c "import sys; import PySide6, PIL, ultralytics, openai, yaml, ai_edge_litert; exit(0 if sys.version_info[:2] in [(3,10),(3,11),(3,12)] else 1)" >nul 2>nul
exit /b %ERRORLEVEL%

:failed
echo.
echo Review the error above, then run this file again.
pause
endlocal
exit /b 1
