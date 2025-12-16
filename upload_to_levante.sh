#!/bin/bash

##################################################
# Upload Files to Levante HPC
##################################################

LEVANTE_USER="g260141"
LEVANTE_HOST="levante.dkrz.de"
REMOTE_DIR="/work/gg0302/g260141/Jingle_D"
LOCAL_DIR="."

echo "=================================================="
echo "Uploading files to Levante"
echo "=================================================="
echo ""
echo "Local: ${LOCAL_DIR}"
echo "Remote: ${LEVANTE_USER}@${LEVANTE_HOST}:${REMOTE_DIR}"
echo ""

echo "Choose what to upload:"
echo "  1) All Python files and scripts (recommended)"
echo "  2) Python files only"
echo "  3) Submit scripts only"
echo "  4) Documentation only"
echo "  5) Everything (code + docs)"
echo ""
read -p "Enter choice [1-5]: " CHOICE

case $CHOICE in
    1)
        echo ""
        echo "Uploading Python files and scripts..."
        rsync -avz --progress \
            --include='*.py' \
            --include='*.sh' \
            --exclude='*' \
            ${LOCAL_DIR}/ \
            ${LEVANTE_USER}@${LEVANTE_HOST}:${REMOTE_DIR}/
        ;;
    
    2)
        echo ""
        echo "Uploading Python files only..."
        rsync -avz --progress \
            complementary_masking.py \
            model_simple_transformer.py \
            train_simple_ddp.py \
            train_simple_worker.py \
            dataset_wav_pairs.py \
            inference_cascade.py \
            test_masking.py \
            ${LEVANTE_USER}@${LEVANTE_HOST}:${REMOTE_DIR}/
        ;;
    
    3)
        echo ""
        echo "Uploading submit scripts..."
        rsync -avz --progress \
            submit_*.sh \
            run_train_direct.sh \
            ${LEVANTE_USER}@${LEVANTE_HOST}:${REMOTE_DIR}/
        ;;
    
    4)
        echo ""
        echo "Uploading documentation..."
        rsync -avz --progress \
            README*.md \
            IMPLEMENTATION_SUMMARY.md \
            CHECKLIST.md \
            ${LEVANTE_USER}@${LEVANTE_HOST}:${REMOTE_DIR}/
        ;;
    
    5)
        echo ""
        echo "Uploading everything..."
        rsync -avz --progress \
            --exclude='__pycache__/' \
            --exclude='*.pyc' \
            --exclude='checkpoints*/' \
            --exclude='logs/' \
            --exclude='outputs*/' \
            --exclude='inference_outputs/' \
            --exclude='mlruns/' \
            --exclude='dataset*/' \
            --exclude='.git/' \
            --exclude='*.pt' \
            --exclude='*.wav' \
            --exclude='*.png' \
            ${LOCAL_DIR}/ \
            ${LEVANTE_USER}@${LEVANTE_HOST}:${REMOTE_DIR}/
        ;;
    
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "Upload complete!"
echo "=================================================="
echo ""
echo "Files uploaded to: ${LEVANTE_USER}@${LEVANTE_HOST}:${REMOTE_DIR}"
echo ""
echo "Next steps on Levante:"
echo "  1. ssh ${LEVANTE_USER}@${LEVANTE_HOST}"
echo "  2. cd ${REMOTE_DIR}"
echo "  3. mkdir -p logs"
echo "  4. sbatch submit_levante.sh"
echo "=================================================="
