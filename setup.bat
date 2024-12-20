@echo off
setlocal enabledelayedexpansion

REM List of common Python installation paths
set "PYTHON_PATHS=C:\Python312;C:\Program Files\Python312;C:\Program Files\Python311;C:\Program Files\Python310;C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312;C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311;C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310"

set "PYTHON_EXE="
for %%p in (%PYTHON_PATHS%) do (
    if exist "%%p\python.exe" (
        set "PYTHON_EXE=%%p\python.exe"
        goto :found_python
    )
)

:not_found_python
echo Python not found in common locations.
echo Please install Python 3.10 or higher from https://www.python.org/downloads/
exit /b 1

:found_python
echo Found Python at: %PYTHON_EXE%
echo Setting up backend...

REM Create and activate virtual environment
"%PYTHON_EXE%" -m venv venv
call venv\Scripts\activate
@REM venv\Scripts\activate

REM Add before pip installations
echo This installation may take a while. If it appears to hang, wait at least 5 minutes before canceling.
echo If you need to cancel, you can safely run setup.bat again to resume from where it left off.

REM Upgrade pip first
python -m pip install --upgrade pip

REM Check for CUDA availability
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo CUDA GPU detected, installing PyTorch with CUDA support...
    pip install --timeout 1000 --retries 5 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo No CUDA GPU detected, installing CPU version of PyTorch...
    pip install --timeout 1000 --retries 5 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

REM Install core dependencies one by one
pip install --timeout 1000 --retries 5 flask==3.0.3
pip install --timeout 1000 --retries 5 flask-cors==5.0.0
pip install --timeout 1000 --retries 5 basicsr==1.4.2
pip install --timeout 1000 --retries 5 numpy==1.26.3
pip install --timeout 1000 --retries 5 opencv-python==4.10.0.84
pip install --timeout 1000 --retries 5 opencv-python-headless==4.10.0.84
pip install --timeout 1000 --retries 5 Pillow==9.5.0
pip install --timeout 1000 --retries 5 pyiqa==0.1.13
pip install --timeout 1000 --retries 5 realesrgan==0.3.0
pip install --timeout 1000 --retries 5 requests
pip install --timeout 1000 --retries 5 tqdm
pip install --timeout 1000 --retries 5 spandrel

pip install --timeout 1000 --retries 5 packaging --upgrade
pip install --timeout 1000 --retries 5 setuptools --upgrade

pip install --timeout 1000 --retries 5 -r backend/requirements.txt

REM Create necessary directories
mkdir backend\temp_uploads 2>nul
mkdir backend\pre_swapped_channels_results 2>nul
mkdir backend\final_results 2>nul
mkdir backend\weights 2>nul
mkdir backend\metric_weights 2>nul

echo Backend setup complete!

echo.
echo Setting up frontend...
cd frontend/react-app
call npm install
cd ../..

echo.
echo Patching/Updating package files...
REM Check for CUDA availability
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo "CUDA GPU detected, hard coding gpu_id to 0 and device to cuda in package files..."
    python package_modifications/update_package.py
    python package_modifications/update_package_cuda/update_package_cuda.py
) else (
    echo No CUDA GPU detected, falling back to cpu...
    python package_modifications/update_package.py
)
echo Package files patched/updated successfully!
echo.

echo.
echo Setup complete!
echo To start up the project, run: run.bat
echo Alternatively, follow these steps:
echo To activate the virtual environment, run: venv\Scripts\activate
echo To start the frontend, run: cd frontend/react-app ^& npm start
echo To start the backend, run: cd backend ^& python app.py
