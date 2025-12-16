#!/bin/bash

#SBATCH --job-name=create_dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --account=gg0302
#SBATCH --partition=compute
#SBATCH --output=logs/dataset.out
#SBATCH --error=logs/dataset.err
#SBATCH --exclusive
#SBATCH --mem=0

# Create Multi-Pattern Training Dataset on CPU
echo "=================================================="
echo "Creating Multi-Pattern Training Dataset"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: 16"
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
echo ""

# Run dataset creation
echo "Starting dataset creation..."
echo "This will process 32GB of music files and may take 1-2 hours..."
echo ""

python create_dataset_multipattern_fixed.py

echo ""
echo "=================================================="
echo "Dataset creation completed!"
echo "End time: $(date)"
echo "=================================================="

# Show dataset info
if [ -f dataset_multipattern/training_set_multipattern_fixed.npy ]; then
    echo ""
    echo "Dataset files created:"
    ls -lh dataset_multipattern/
    echo ""
    echo "Ready for training! Submit with: sbatch run_levante.sh"
else
    echo ""
    echo "ERROR: Dataset files not found!"
    echo "Check logs/dataset-${SLURM_JOB_ID}.err for errors"
fi
