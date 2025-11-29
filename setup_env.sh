#!/bin/bash
# Environment Setup Script for Multi-Pattern Audio Fusion
# Run this on HPC Levante after uploading your code

set -e  # Exit on error

echo "=================================================="
echo "Multi-Pattern Fusion - Environment Setup"
echo "=================================================="

# Configuration
ENV_NAME="multipattern_env"
PYTHON_VERSION="3.10"

echo ""
echo "Step 1: Creating Python virtual environment..."
echo "Environment name: $ENV_NAME"
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
python${PYTHON_VERSION} -m venv $ENV_NAME

# Activate environment
source $ENV_NAME/bin/activate

echo ""
echo "Step 2: Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo ""
echo "Step 3: Installing PyTorch with CUDA support..."
# Install PyTorch with CUDA 11.8 (adjust based on your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Step 4: Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "Step 5: Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

echo ""
echo "Step 6: Testing imports..."
python -c "import librosa; import numpy; import scipy; import soundfile; import tqdm; print('All packages imported successfully!')"

echo ""
echo "=================================================="
echo "Environment setup complete!"
echo "=================================================="
echo "Activate with: source $ENV_NAME/bin/activate"
echo "Python executable: $(which python)"
echo ""
echo "Next steps:"
echo "  1. Generate dataset: python create_dataset_multipattern.py"
echo "  2. Submit training: sbatch run_levante.sh"
echo "=================================================="
