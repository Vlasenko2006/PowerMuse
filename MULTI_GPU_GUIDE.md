# Multi-GPU Training Setup

## Files Added

1. **main_multipattern_ddp.py** - DistributedDataParallel version of training script
2. **run_levante.sh** - SLURM job script for HPC Levante

## Usage

### Local Testing (Single GPU)
```bash
python main_multipattern_ddp.py
```

### Local Multi-GPU (e.g., 4 GPUs)
```bash
torchrun --nproc_per_node=4 main_multipattern_ddp.py
```

### HPC Levante (4 GPUs, SLURM)
```bash
# Make script executable
chmod +x run_levante.sh

# Submit job
sbatch run_levante.sh

# Monitor job
squeue -u g260141

# Check output
tail -f logs/train-<job_id>.out
```

## Key Changes for DDP

### 1. Distributed Setup
- Automatically detects SLURM environment variables
- Initializes process groups with NCCL backend for GPU
- Each GPU gets assigned to a separate process

### 2. Data Loading
- Uses `DistributedSampler` to split data across GPUs
- Each GPU processes different batches
- Automatic gradient synchronization across GPUs

### 3. Model Wrapping
- Model wrapped with `DDP` for synchronized training
- `find_unused_parameters=True` for 3-phase training with frozen layers
- Gradients averaged across all GPUs automatically

### 4. Effective Batch Size
- Per-GPU batch size: 4
- Total GPUs: 4
- Accumulation steps: 16
- **Effective batch size: 4 × 4 × 16 = 256**

### 5. Checkpointing
- Only rank 0 saves checkpoints
- All ranks can load from checkpoint
- Synchronized epoch tracking across all processes

## Performance Benefits

**Single GPU:**
- Batch size: 4
- Effective batch size: 64 (with accumulation)
- Estimated time per epoch: ~2 hours

**4 GPUs with DDP:**
- Per-GPU batch size: 4
- Effective batch size: 256 (4 × 4 × 16)
- Estimated time per epoch: ~30 minutes
- **~4x speedup with near-linear scaling**

## Monitoring

Check training progress:
```bash
# Real-time output
tail -f logs/train-<job_id>.out

# Check GPU usage
srun --jobid=<job_id> nvidia-smi

# Check errors
tail -f logs/train-<job_id>.err
```

## Next Steps

1. **Upload code to Levante:**
   ```bash
   rsync -avz --partial --progress --timeout=300 \
     /Users/andreyvlasenko/tst/Jingle/ \
     g260141@levante.dkrz.de:/work/gg0302/g260141/Jingle/
   ```

2. **Verify Python environment has required packages:**
   ```bash
   ssh g260141@levante.dkrz.de
   /work/gg0302/g260141/BART/bart_env/bin/python3.10 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```

3. **Generate dataset on Levante:**
   ```bash
   cd /work/gg0302/g260141/Jingle
   /work/gg0302/g260141/BART/bart_env/bin/python3.10 create_dataset_multipattern.py
   ```

4. **Submit training job:**
   ```bash
   sbatch run_levante.sh
   ```

## Troubleshooting

**Issue: "ImportError: No module named torch.distributed"**
- Solution: Ensure PyTorch was compiled with distributed support
- Check: `python -c "import torch.distributed as dist; print(dist.is_available())"`

**Issue: "NCCL error: unhandled system error"**
- Solution: Try setting `export NCCL_IB_DISABLE=1` in run_levante.sh

**Issue: "Rank X timeout waiting for other ranks"**
- Solution: Check that all GPUs are accessible and --ntasks-per-node matches --gpus-per-node

**Issue: Different GPUs have different data**
- This is expected! DistributedSampler splits data across GPUs
- Gradients are automatically synchronized during backward pass
