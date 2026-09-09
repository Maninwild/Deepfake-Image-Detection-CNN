@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher was not found. Install 64-bit Python 3.12 and try again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python 3.12 environment...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Python 3.12 is required. Install it and select Add Python to PATH.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed

if not exist "raw\real" (
    python code\extract_dataset.py
    if errorlevel 1 goto :failed
)

if not exist "arrays\X_test.npy" (
    python code\validate_dataset.py
    if errorlevel 1 goto :failed
    python code\split_data.py
    if errorlevel 1 goto :failed
    python code\preprocess_arrays.py
    if errorlevel 1 goto :failed
)

python code\evaluate.py
if errorlevel 1 goto :failed

echo Opening the prediction interface...
python -m streamlit run app.py
exit /b 0

:failed
echo.
echo Project setup failed. Read README.md for troubleshooting.
pause
exit /b 1
