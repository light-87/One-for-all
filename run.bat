@echo off
REM run.bat - Windows execution script for One-For-All Phosphorylation Prediction Pipeline
REM Usage: run.bat [options]

echo ==========================================
echo One-For-All Phosphorylation Prediction Pipeline
echo ==========================================

REM Set environment variables for optimization
set CUDA_LAUNCH_BLOCKING=0
set OMP_NUM_THREADS=4
set TOKENIZERS_PARALLELISM=false

echo Using current Python environment...

REM Check for required data files
echo Checking for required data files...
set missing_files=0

if exist "data\Sequence_data.txt" (
    echo ✓ data\Sequence_data.txt
) else (
    echo ✗ data\Sequence_data.txt - MISSING
    set missing_files=1
)

if exist "data\labels.xlsx" (
    echo ✓ data\labels.xlsx
) else (
    echo ✗ data\labels.xlsx - MISSING
    set missing_files=1
)

if exist "data\physiochemical_property.csv" (
    echo ✓ data\physiochemical_property.csv
) else (
    echo ✗ data\physiochemical_property.csv - MISSING
    set missing_files=1
)

if %missing_files% EQU 1 (
    echo.
    echo Error: Missing required data files.
    echo Please ensure all data files are in the data\ directory.
    pause
    exit /b 1
)

REM Check GPU availability
echo Checking GPU availability...
python -c "import torch; print(f'✓ CUDA available with {torch.cuda.device_count()} GPU(s)') if torch.cuda.is_available() else print('CPU mode - CUDA not available'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

echo Starting pipeline execution...
echo Log file will be saved to outputs\experiment.log
echo.

REM Create outputs directory if it doesn't exist
if not exist "outputs" mkdir outputs

REM Record start time
echo Pipeline started at: %DATE% %TIME%

REM Parse command line arguments (basic implementation)
set PYTHON_ARGS=
set SHOW_HELP=0

:parse_args
if "%~1"=="" goto run_pipeline
if "%~1"=="--config" (
    set PYTHON_ARGS=%PYTHON_ARGS% --config %2
    shift
    shift
    goto parse_args
)
if "%~1"=="--skip-preprocessing" (
    set PYTHON_ARGS=%PYTHON_ARGS% --skip-preprocessing
    shift
    goto parse_args
)
if "%~1"=="--skip-features" (
    set PYTHON_ARGS=%PYTHON_ARGS% --skip-features
    shift
    goto parse_args
)
if "%~1"=="--skip-xgboost" (
    set PYTHON_ARGS=%PYTHON_ARGS% --skip-xgboost
    shift
    goto parse_args
)
if "%~1"=="--skip-transformers" (
    set PYTHON_ARGS=%PYTHON_ARGS% --skip-transformers
    shift
    goto parse_args
)
if "%~1"=="--skip-ensemble" (
    set PYTHON_ARGS=%PYTHON_ARGS% --skip-ensemble
    shift
    goto parse_args
)
if "%~1"=="--cross-validation" (
    set PYTHON_ARGS=%PYTHON_ARGS% --cross-validation
    shift
    goto parse_args
)
if "%~1"=="--help" (
    echo Usage: %0 [options]
    echo.
    echo Options:
    echo   --config FILE             Use custom config file (default: config.yaml)
    echo   --skip-preprocessing      Skip data preprocessing step
    echo   --skip-features          Skip feature extraction step
    echo   --skip-xgboost           Skip XGBoost training step
    echo   --skip-transformers      Skip transformer training step
    echo   --skip-ensemble          Skip ensemble training step
    echo   --cross-validation       Run cross-validation
    echo   --help                   Show this help message
    pause
    exit /b 0
)
echo Unknown option: %~1
echo Use --help for usage information
pause
exit /b 1

:run_pipeline
REM Run the main pipeline
python one_for_all.py %PYTHON_ARGS%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==========================================
    echo Pipeline completed successfully!
    echo Completed at: %DATE% %TIME%
    echo ==========================================
    echo.
    echo Results saved to outputs\ directory
    echo Model checkpoints saved to checkpoints\ directory
    echo.
    echo Next steps:
    echo 1. Check outputs\experiment.log for detailed logs
    echo 2. Review evaluation metrics in outputs\
    echo 3. Examine model checkpoints in checkpoints\
    echo 4. Use trained models for prediction on new data
) else (
    echo.
    echo ==========================================
    echo Pipeline failed with error code: %ERRORLEVEL%
    echo Failed at: %DATE% %TIME%
    echo ==========================================
    echo.
    echo Check outputs\experiment.log for error details
    echo Common issues:
    echo 1. Insufficient GPU memory - try reducing batch sizes
    echo 2. Missing dependencies - install required packages
    echo 3. Corrupted data files - verify data integrity
    echo 4. Configuration errors - check config.yaml syntax
)

pause