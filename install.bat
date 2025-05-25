@echo off
REM install.bat - Windows installation script for One-For-All Phosphorylation Prediction Pipeline

echo ==========================================
echo One-For-All Pipeline - Installation Script
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo ✓ Python %python_version% found

REM Check if pip is installed
pip --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: pip is not installed
    echo Please install pip or reinstall Python with pip included
    pause
    exit /b 1
)

echo ✓ pip found

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing core dependencies...

REM Install core scientific computing libraries
echo Installing numpy...
pip install "numpy>=1.21.0"
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to install numpy
    goto :error
)
echo ✓ numpy installed

echo Installing pandas...
pip install "pandas>=1.3.0"
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to install pandas
    goto :error
)
echo ✓ pandas installed

echo Installing scipy...
pip install "scipy>=1.7.0"
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to install scipy
    goto :error
)
echo ✓ scipy installed

echo Installing scikit-learn...
pip install "scikit-learn>=1.0.0"
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to install scikit-learn
    goto :error
)
echo ✓ scikit-learn installed

echo Installing matplotlib...
pip install "matplotlib>=3.5.0"
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to install matplotlib
    goto :error
)
echo ✓ matplotlib installed

echo Installing seaborn...
pip install "seaborn>=0.11.0"
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to install seaborn
    goto :error
)
echo ✓ seaborn installed

echo Installing tqdm...
pip install "tqdm>=4.64.0"
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to install tqdm
    goto :error
)
echo ✓ tqdm installed

echo Installing pyyaml...
pip install "pyyaml>=6.0"
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to install pyyaml
    goto :error
)
echo ✓ pyyaml installed

echo Installing openpyxl...
pip install "openpyxl>=3.0.0"
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to install openpyxl
    goto :error
)
echo ✓ openpyxl installed

echo Installing psutil...
pip install "psutil>=5.8.0"
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to install psutil
    goto :error
)
echo ✓ psutil installed

echo.
echo Installing machine learning libraries...

echo Installing XGBoost...
pip install "xgboost>=1.6.0"
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to install XGBoost
    goto :error
)
echo ✓ XGBoost installed

echo Installing Optuna...
pip install "optuna>=3.0.0"
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to install Optuna
    goto :error
)
echo ✓ Optuna installed

echo.
echo Installing datatable...
pip install "datatable>=1.0.0"
if %ERRORLEVEL% NEQ 0 (
    echo ! datatable installation failed, trying alternative method...
    pip install datatable --no-cache-dir
    if %ERRORLEVEL% NEQ 0 (
        echo ✗ datatable installation failed - you may need to install manually
        echo   Try: pip install datatable --no-binary :all:
    ) else (
        echo ✓ datatable installed
    )
) else (
    echo ✓ datatable installed
)

echo.
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ NVIDIA GPU detected
    set cuda_available=true
) else (
    echo ! No NVIDIA GPU detected, will install CPU-only versions
    set cuda_available=false
)

echo.
echo Installing PyTorch and transformers...

if "%cuda_available%"=="true" (
    echo Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo Installing CPU-only PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to install PyTorch
    goto :error
)
echo ✓ PyTorch installed

echo Installing transformers...
pip install "transformers>=4.20.0"
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to install transformers
    goto :error
)
echo ✓ transformers installed

echo Installing tokenizers...
pip install "tokenizers>=0.12.0"
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Failed to install tokenizers
    goto :error
)
echo ✓ tokenizers installed

echo Installing accelerate...
pip install "accelerate>=0.20.0"
if %ERRORLEVEL% NEQ 0 (
    echo ! accelerate installation failed (optional)
) else (
    echo ✓ accelerate installed
)

echo.
echo Installing logging and monitoring tools...

echo Installing wandb...
pip install "wandb>=0.13.0"
if %ERRORLEVEL% NEQ 0 (
    echo ! wandb installation failed (optional)
) else (
    echo ✓ wandb installed
)

echo Installing tensorboard...
pip install "tensorboard>=2.9.0"
if %ERRORLEVEL% NEQ 0 (
    echo ! tensorboard installation failed (optional)
) else (
    echo ✓ tensorboard installed
)

echo Installing tensorboardX...
pip install "tensorboardX>=2.5.1"
if %ERRORLEVEL% NEQ 0 (
    echo ! tensorboardX installation failed (optional)
) else (
    echo ✓ tensorboardX installed
)

echo.
echo Installing interpretability tools...

echo Installing SHAP...
pip install "shap>=0.41.0"
if %ERRORLEVEL% NEQ 0 (
    echo ! SHAP installation failed - you can install it manually later if needed
    echo   Try: pip install shap --no-cache-dir
) else (
    echo ✓ SHAP installed
)

echo.
echo Installing optional development tools...

echo Installing Jupyter...
pip install "jupyter>=1.0.0"
if %ERRORLEVEL% NEQ 0 (
    echo ! Jupyter installation failed (optional)
) else (
    echo ✓ Jupyter installed
)

echo Installing plotly...
pip install "plotly>=5.0.0"
if %ERRORLEVEL% NEQ 0 (
    echo ! plotly installation failed (optional)
) else (
    echo ✓ plotly installed
)

echo Installing memory-profiler...
pip install "memory-profiler>=0.60.0"
if %ERRORLEVEL% NEQ 0 (
    echo ! memory-profiler installation failed (optional)
) else (
    echo ✓ memory-profiler installed
)

echo.
echo ==========================================
echo Installation Summary
echo ==========================================

echo Verifying installations...

python -c "import sys; packages_to_check = ['numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn', 'torch', 'transformers', 'xgboost', 'tqdm', 'yaml']; success = True; [print(f'✓ {package}') if __import__(package) or True else (print(f'✗ {package} - FAILED'), setattr(sys.modules[__name__], 'success', False)) for package in packages_to_check]; optional_packages = ['datatable', 'shap', 'wandb']; [print(f'✓ {package} (optional)') if __import__(package) or True else print(f'! {package} (optional) - not available') for package in optional_packages]; import torch; print(f'✓ CUDA available: {torch.cuda.device_count()} GPU(s)') if torch.cuda.is_available() else print('! CUDA not available - CPU mode only'); print('\n🎉 Core installation successful!\nYou can now run the pipeline with: run.bat') if success else (print('\n⚠️  Some core packages failed to install.\nPlease check the errors above and install missing packages manually.'), sys.exit(1))"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==========================================
    echo Installation completed successfully!
    echo ==========================================
    echo.
    echo Next steps:
    echo 1. Place your data files in the data\ directory:
    echo    - data\Sequence_data.txt
    echo    - data\labels.xlsx
    echo    - data\physiochemical_property.csv
    echo.
    echo 2. Run the pipeline:
    echo    run.bat
    echo.
    echo 3. Or test the installation:
    echo    python -c "import torch, transformers, xgboost; print('All core libraries imported successfully!')"
    echo.
    echo For more information, see README.md
    goto :success
) else (
    goto :error
)

:error
echo.
echo Installation completed with some errors.
echo Please check the output above and install missing packages manually.
echo.
echo Common solutions:
echo 1. Run as administrator
echo 2. Update pip: python -m pip install --upgrade pip
echo 3. Install Visual C++ Build Tools for Windows
echo 4. Try installing packages one by one manually
pause
exit /b 1

:success
pause
exit /b 0