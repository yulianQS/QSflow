@echo off
setlocal EnableExtensions
title QSflow - GPU Setup
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Project .venv does not exist. Run the main launcher first.
    pause
    exit /b 1
)

echo Installing CUDA 12.6 PyTorch for NVIDIA GPU...
echo This download is large and may take several minutes.
".venv\Scripts\python.exe" -m pip install --upgrade --force-reinstall torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto :failed

echo Verifying CUDA...
".venv\Scripts\python.exe" -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('GPU READY:', torch.cuda.get_device_name(0), 'CUDA', torch.version.cuda)"
if errorlevel 1 goto :failed

echo GPU environment is ready. Restart QSflow.
pause
exit /b 0

:failed
echo GPU setup failed. Review the message above.
pause
exit /b 1
