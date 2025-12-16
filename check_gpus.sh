#!/bin/bash

# GPU diagnostic script
echo "=================================================="
echo "GPU Diagnostic Check"
echo "=================================================="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo ""

echo "1. Checking nvidia-smi..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "⚠️  nvidia-smi not found!"
fi
echo ""

echo "2. Checking CUDA_VISIBLE_DEVICES..."
echo "CUDA_VISIBLE_DEVICES = '$CUDA_VISIBLE_DEVICES'"
echo ""

echo "3. Checking SLURM GPU allocation..."
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "SLURM_GPUS = $SLURM_GPUS"
echo "SLURM_GPUS_ON_NODE = $SLURM_GPUS_ON_NODE"
echo "SLURM_JOB_GPUS = $SLURM_JOB_GPUS"
echo ""

echo "4. Checking PyTorch GPU detection..."
python << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("⚠️  No GPUs detected by PyTorch!")
EOF
echo ""

echo "5. Checking /proc/driver/nvidia/gpus..."
if [ -d "/proc/driver/nvidia/gpus" ]; then
    ls -la /proc/driver/nvidia/gpus/
else
    echo "⚠️  /proc/driver/nvidia/gpus not found!"
fi
echo ""

echo "=================================================="
echo "Diagnostic complete"
echo "=================================================="
