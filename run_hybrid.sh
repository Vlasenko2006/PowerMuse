#!/bin/bash
# Wrapper script that always pulls latest changes first
# This ensures we get the latest run_train_hybrid.sh script

cd /work/gg0302/g260141/Jingle_D

echo "=================================================="
echo "Pulling latest code from adaptive-window-selection branch..."
echo "=================================================="
git pull origin adaptive-window-selection
echo ""

echo "Running training script..."
bash run_train_hybrid.sh
