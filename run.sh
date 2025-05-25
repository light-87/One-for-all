#!/bin/bash

# run.sh - Execution script for One-For-All Phosphorylation Prediction Pipeline
# Usage: ./run.sh [options]

echo "=========================================="
echo "One-For-All Phosphorylation Prediction Pipeline"
echo "=========================================="

# Set environment variables for optimization
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# Check if conda environment exists
if command -v conda &> /dev/null; then
    echo "Conda found, checking for phospho_env environment..."
    if conda env list | grep -q "phospho_env"; then
        echo "Activating conda environment: phospho_env"
        source activate phospho_env
    else
        echo "Warning: phospho_env environment not found. Using current environment."
    fi
else
    echo "Conda not found. Using current Python environment."
fi

# Check Python and required packages
echo "Checking Python installation..."
python --version

echo "Checking required packages..."
python -c "
import sys
required_packages = [
    'numpy', 'pandas', 'datatable', 'torch', 'transformers', 
    'xgboost', 'sklearn', 'matplotlib', 'seaborn', 'tqdm', 'yaml'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f'✓ {package}')
    except ImportError:
        missing_packages.append(package)
        print(f'✗ {package} - MISSING')

if missing_packages:
    print(f'\nError: Missing packages: {missing_packages}')
    print('Please install missing packages before running the pipeline.')
    sys.exit(1)
else:
    print('\nAll required packages found!')
"

# Check if the import check failed
if [ $? -ne 0 ]; then
    echo "Package check failed. Please install missing packages."
    exit 1
fi

# Check for required data files
echo "Checking for required data files..."
data_files=(
    "data/Sequence_data.txt"
    "data/labels.xlsx" 
    "data/physiochemical_property.csv"
)

missing_files=()
for file in "${data_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file - MISSING"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo ""
    echo "Error: Missing required data files:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "Please ensure all data files are in the correct locations."
    exit 1
fi

# Check available GPU memory if CUDA is available
echo "Checking GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f'✓ CUDA available with {gpu_count} GPU(s)')
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f'  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)')
else:
    print('CPU mode - CUDA not available')
"

echo "Starting pipeline execution..."
echo "Log file will be saved to outputs/experiment.log"
echo ""

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Run the pipeline with error handling
start_time=$(date)
echo "Pipeline started at: $start_time"

# Parse command line arguments
PYTHON_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            PYTHON_ARGS="$PYTHON_ARGS --config $2"
            shift 2
            ;;
        --skip-preprocessing)
            PYTHON_ARGS="$PYTHON_ARGS --skip-preprocessing"
            shift
            ;;
        --skip-features)
            PYTHON_ARGS="$PYTHON_ARGS --skip-features"
            shift
            ;;
        --skip-xgboost)
            PYTHON_ARGS="$PYTHON_ARGS --skip-xgboost"
            shift
            ;;
        --skip-transformers)
            PYTHON_ARGS="$PYTHON_ARGS --skip-transformers"
            shift
            ;;
        --skip-ensemble)
            PYTHON_ARGS="$PYTHON_ARGS --skip-ensemble"
            shift
            ;;
        --cross-validation)
            PYTHON_ARGS="$PYTHON_ARGS --cross-validation"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --config FILE             Use custom config file (default: config.yaml)"
            echo "  --skip-preprocessing      Skip data preprocessing step"
            echo "  --skip-features          Skip feature extraction step"
            echo "  --skip-xgboost           Skip XGBoost training step"
            echo "  --skip-transformers      Skip transformer training step"
            echo "  --skip-ensemble          Skip ensemble training step"
            echo "  --cross-validation       Run cross-validation"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run the main pipeline
if python one_for_all.py $PYTHON_ARGS; then
    end_time=$(date)
    echo ""
    echo "=========================================="
    echo "Pipeline completed successfully!"
    echo "Started:   $start_time"
    echo "Completed: $end_time"
    echo "=========================================="
    echo ""
    echo "Results saved to outputs/ directory"
    echo "Model checkpoints saved to checkpoints/ directory"
    echo ""
    echo "Next steps:"
    echo "1. Check outputs/experiment.log for detailed logs"
    echo "2. Review evaluation metrics in outputs/"
    echo "3. Examine model checkpoints in checkpoints/"
    echo "4. Use trained models for prediction on new data"
else
    error_code=$?
    end_time=$(date)
    echo ""
    echo "=========================================="
    echo "Pipeline failed with error code: $error_code"
    echo "Started:   $start_time"
    echo "Failed:    $end_time"
    echo "=========================================="
    echo ""
    echo "Check outputs/experiment.log for error details"
    echo "Common issues:"
    echo "1. Insufficient GPU memory - try reducing batch sizes"
    echo "2. Missing dependencies - install required packages"
    echo "3. Corrupted data files - verify data integrity"
    echo "4. Configuration errors - check config.yaml syntax"
    exit $error_code
fi