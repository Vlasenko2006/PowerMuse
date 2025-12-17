#!/bin/bash
# Sync updated files to remote server

REMOTE_USER="g260141"
REMOTE_HOST="levante.dkrz.de"
REMOTE_DIR="/work/gg0302/g260141/Jingle_D"

echo "=================================================="
echo "Syncing updated files to remote server"
echo "=================================================="
echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
echo ""

# List of files to sync
FILES=(
    "model_simple_transformer.py"
    "creative_agent.py"
    "compositional_creative_agent.py"
    "correlation_penalty.py"
    "train_simple_worker.py"
    "train_simple_ddp.py"
    "run_train_creative_agent_fixed.sh"
    "run_train_creative_agent_push_complementarity.sh"
    "training/losses.py"
)

echo "Files to sync:"
for file in "${FILES[@]}"; do
    echo "  - $file"
done
echo ""

# Sync each file
for file in "${FILES[@]}"; do
    echo "Syncing: $file"
    # Preserve directory structure for files with paths
    if [[ "$file" == */* ]]; then
        # File has a directory path, create remote dir first
        remote_dir="${REMOTE_DIR}/$(dirname "$file")"
        ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${remote_dir}"
        scp "$file" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${file}"
    else
        # Regular file, sync to root
        scp "$file" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
    fi
    if [ $? -eq 0 ]; then
        echo "  ✓ $file synced successfully"
    else
        echo "  ✗ Failed to sync $file"
        exit 1
    fi
done

echo ""
echo "=================================================="
echo "✅ All files synced successfully!"
echo "=================================================="
echo ""
echo "Now on the remote server, run:"
echo "  bash run_train_creative_agent_fixed.sh"
echo ""
echo "Note: Make sure to chmod +x run_train_creative_agent_fixed.sh on remote"
echo ""
