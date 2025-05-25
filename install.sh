#!/bin/bash

# install.sh - Installation script for One-For-All Phosphorylation Prediction Pipeline

echo "=========================================="
echo "One-For-All Pipeline - Installation Script"
echo "=========================================="

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    echo "Please install Python 3.8+ before running this script"
    exit 1
fi

# Check Python version
python_version=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
required_version="3.8"

if ! python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "Error: Python $python_version found, but Python $required_version+ is required"
    exit 1
fi

echo "✓ Python $python_version found"

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed"
    echo "Please install pip before running this script"
    exit 1
fi

echo "✓ pip found"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Function to install packages with error handling
install_package() {
    package=$1
    echo "Installing $package..."
    if pip install "$package"; then
        echo "✓ $package installed successfully"
    else
        echo "✗ Failed to install $package"
        return 1
    fi
}

# Check for CUDA availability
echo "Checking for CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    cuda_available=true
else
    echo "! No NVIDIA GPU detected, will install CPU-only versions"
    cuda_available=false
fi

echo ""
echo "Installing core dependencies..."

# Install core scientific computing libraries
core_packages=(
    "numpy>=1.21.0"
    "pandas>=1.3.0"
    "scipy>=1.7.0"
    "scikit-learn>=1.0.0"
    "matplotlib>=3.5.0"
    "seaborn>=0.11.0"
    "tqdm>=4.64.0"
    "pyyaml>=6.0"
    "openpyxl>=3.0.0"
    "psutil>=5.8.0"
    "joblib>=1.1.0"
)

for package in "${core_packages[@]}"; do
    install_package "$package" || exit 1
done

echo ""
echo "Installing machine learning libraries..."

# Install ML libraries
ml_packages=(
    "xgboost>=1.6.0"
    "optuna>=3.0.0"
)

for package in "${ml_packages[@]}"; do
    install_package "$package" || exit 1
done

echo ""
echo "Installing datatable..."
# Datatable might need special handling
if pip install datatable>=1.0.0; then
    echo "✓ datatable installed successfully"
else
    echo "! datatable installation failed, trying alternative method..."
    # Try installing from source or alternative
    pip install https://github.com/h2oai/datatable/releases/download/v1.0.0/datatable-1.0.0-cp39-cp39-linux_x86_64.whl || \
    pip install datatable --no-cache-dir || \
    echo "✗ datatable installation failed - you may need to install manually"
fi

echo ""
echo "Installing PyTorch and transformers..."

# Install PyTorch based on CUDA availability
if [ "$cuda_available" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    # Check CUDA version
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        echo "CUDA version: $cuda_version"
        
        if [[ "$cuda_version" == "11.8"* ]]; then
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        elif [[ "$cuda_version" == "12."* ]]; then
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        else
            echo "Installing default CUDA version..."
            pip install torch torchvision torchaudio
        fi
    else
        echo "Installing default PyTorch with CUDA..."
        pip install torch torchvision torchaudio
    fi
else
    echo "Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install transformers and related
transformers_packages=(
    "transformers>=4.20.0"
    "tokenizers>=0.12.0"
    "accelerate>=0.20.0"
)

for package in "${transformers_packages[@]}"; do
    install_package "$package" || exit 1
done

echo ""
echo "Installing logging and monitoring tools..."

# Install logging/monitoring
logging_packages=(
    "wandb>=0.13.0"
    "tensorboard>=2.9.0"
    "tensorboardX>=2.5.1"
)

for package in "${logging_packages[@]}"; do
    install_package "$package" || exit 1
done

echo ""
echo "Installing interpretability tools..."

# Install SHAP (might need compilation)
if pip install shap>=0.41.0; then
    echo "✓ SHAP installed successfully"
else
    echo "! SHAP installation failed - you can install it manually later if needed"
    echo "  Try: pip install shap --no-cache-dir"
fi

echo ""
echo "Installing optional development tools..."

# Optional packages
optional_packages=(
    "jupyter>=1.0.0"
    "ipykernel>=6.0.0"
    "notebook>=6.4.0"
    "plotly>=5.0.0"
    "memory-profiler>=0.60.0"
    "progressbar2>=4.0.0"
)

for package in "${optional_packages[@]}"; do
    if install_package "$package"; then
        echo "  ✓ Optional: $package"
    else
        echo "  ! Optional: $package failed (not critical)"
    fi
done

echo ""
echo "=========================================="
echo "Installation Summary"
echo "=========================================="

# Verify key installations
echo "Verifying installations..."

python -c "
import sys
packages_to_check = [
    'numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn',
    'torch', 'transformers', 'xgboost', 'tqdm', 'yaml'
]

success = True
for package in packages_to_check:
    try:
        __import__(package)
        print(f'✓ {package}')
    except ImportError:
        print(f'✗ {package} - FAILED')
        success = False

# Check optional packages
optional_packages = ['datatable', 'shap', 'wandb']
for package in optional_packages:
    try:
        __import__(package)
        print(f'✓ {package} (optional)')
    except ImportError:
        print(f'! {package} (optional) - not available')

# Check CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f'✓ CUDA available: {torch.cuda.device_count()} GPU(s)')
    else:
        print('! CUDA not available - CPU mode only')
except:
    print('✗ PyTorch CUDA check failed')

if success:
    print('\n🎉 Core installation successful!')
    print('You can now run the pipeline with: ./run.sh')
else:
    print('\n⚠️  Some core packages failed to install.')
    print('Please check the errors above and install missing packages manually.')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Installation completed successfully!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Place your data files in the data/ directory:"
    echo "   - data/Sequence_data.txt"
    echo "   - data/labels.xlsx" 
    echo "   - data/physiochemical_property.csv"
    echo ""
    echo "2. Run the pipeline:"
    echo "   ./run.sh"
    echo ""
    echo "3. Or test the installation:"
    echo "   python -c 'import torch, transformers, xgboost; print(\"All core libraries imported successfully!\")'"
    echo ""
    echo "For more information, see README.md"
else
    echo ""
    echo "Installation completed with some errors."
    echo "Please check the output above and install missing packages manually."
    exit 1
fi