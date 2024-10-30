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

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install both CPU and CUDA versions of PyTorch
echo Installing CPU version of PyTorch...
pip install --timeout 1000 --retries 5 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo Installing CUDA version of PyTorch...
pip install --timeout 1000 --retries 5 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Install core dependencies
pip install --timeout 1000 --retries 5 flask flask-cors Pillow numpy opencv-python

REM Install pyiqa and its dependencies separately with increased timeout
pip install --timeout 1000 --retries 5 basicsr
pip install --timeout 1000 --retries 5 realesrgan
pip install --timeout 1000 --retries 5 pyiqa

REM Then install the rest
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
echo Setup complete!
echo To start up the project, run: run.bat
echo Alternatively, follow these steps:
echo To activate the virtual environment, run: venv\Scripts\activate
echo To start the frontend, run: cd frontend/react-app ^& npm start
echo To start the backend, run: cd backend ^& python app.py
