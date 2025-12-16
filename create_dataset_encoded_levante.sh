#!/bin/bash

#SBATCH --job-name=create_encoded_dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --account=gg0302
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dataset_encoded.out
#SBATCH --error=logs/dataset_encoded.err
#SBATCH --mem=64G

# Create Encoded Multi-Pattern Training Dataset with EnCodec
echo "=================================================="
echo "Creating Encoded Multi-Pattern Training Dataset"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: 16"
echo "GPU: 1"
echo "Memory: 64G"
echo "Start time: $(date)"
echo "=================================================="

# Navigate to project directory
cd /work/gg0302/g260141/Jingle

# Create logs directory
mkdir -p logs

# Activate environment
source multipattern_env/bin/activate

echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  Working directory: $(pwd)"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Check GPU availability
nvidia-smi

# Run encoded dataset creation
echo ""
echo "Starting encoded dataset creation..."
echo "This will encode audio chunks with EnCodec (320x compression)"
echo "Expected time: 2-4 hours depending on GPU..."
echo ""

python dataset_encoded_expanded.py #create_dataset_multipattern_encoded.py

echo ""
echo "=================================================="
echo "Encoded dataset creation completed!"
echo "End time: $(date)"
echo "=================================================="

# Show dataset info
if [ -f dataset_multipattern_encoded/train_inputs.npy ]; then
    echo ""
    echo "Encoded dataset files created:"
    ls -lh dataset_multipattern_encoded/
    echo ""
    echo "Dataset shape information:"
    python -c "
import numpy as np
train_inputs = np.load('dataset_multipattern_encoded/train_inputs.npy', mmap_mode='r')
train_targets = np.load('dataset_multipattern_encoded/train_targets.npy', mmap_mode='r')
val_inputs = np.load('dataset_multipattern_encoded/val_inputs.npy', mmap_mode='r')
val_targets = np.load('dataset_multipattern_encoded/val_targets.npy', mmap_mode='r')
print(f'Training: {train_inputs.shape} inputs, {train_targets.shape} targets')
print(f'Validation: {val_inputs.shape} inputs, {val_targets.shape} targets')
print(f'Encoding dim: {train_inputs.shape[2]}')
print(f'Encoded frames: {train_inputs.shape[3]}')
"
    echo ""
    echo "Ready for transformer training!"
else
    echo ""
    echo "ERROR: Encoded dataset files not found!"
    echo "Check logs/dataset_encoded.err for errors"
fi
