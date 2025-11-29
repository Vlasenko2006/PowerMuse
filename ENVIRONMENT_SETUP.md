# Environment Setup Instructions

## Quick Setup on HPC Levante

### 1. Upload code to Levante
```bash
# From your local machine
rsync -avz --partial --progress --timeout=300 \
  /Users/andreyvlasenko/tst/Jingle/ \
  g260141@levante.dkrz.de:/work/gg0302/g260141/Jingle/
```

### 2. SSH to Levante
```bash
ssh g260141@levante.dkrz.de
cd /work/gg0302/g260141/Jingle
```

### 3. Run setup script
```bash
chmod +x setup_env.sh
./setup_env.sh
```

This creates a virtual environment named `multipattern_env` with all required packages.

---

## Manual Setup (if script fails)

### Option A: Create New Environment

```bash
# Create virtual environment
python3.10 -m venv multipattern_env

# Activate
source multipattern_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other packages
pip install -r requirements.txt

# Verify
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Option B: Use Existing BART Environment

If your BART environment already has PyTorch with CUDA:

```bash
# Activate BART environment
source /work/gg0302/g260141/BART/bart_env/bin/activate

# Install missing packages
pip install librosa soundfile pydub

# Verify
python -c "import torch, librosa, soundfile, pydub; print('All packages available')"
```

Then edit `run_levante.sh` line 26 to use BART environment:
```bash
# Comment out:
# source multipattern_env/bin/activate

# Uncomment:
source /work/gg0302/g260141/BART/bart_env/bin/activate
```

---

## Verify Installation

```bash
# Activate environment
source multipattern_env/bin/activate

# Check PyTorch + CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

# Check audio libraries
python -c "import librosa, soundfile, pydub; print('Audio libraries OK')"

# Check distributed support
python -c "import torch.distributed as dist; print(f'Distributed available: {dist.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA: True
GPUs: 4
Audio libraries OK
Distributed available: True
```

---

## Python Requirements

**requirements.txt** includes:
- `torch>=2.0.0` - Deep learning framework
- `torchaudio>=2.0.0` - Audio processing for PyTorch
- `numpy>=1.23.0` - Numerical computing
- `scipy>=1.10.0` - Scientific computing
- `librosa>=0.10.0` - Audio analysis
- `soundfile>=0.12.0` - Audio file I/O
- `pydub>=0.25.0` - Audio manipulation
- `tqdm>=4.65.0` - Progress bars

---

## CUDA Version Check

If PyTorch installation fails, check your CUDA version:

```bash
# Check CUDA version
nvcc --version

# Or
nvidia-smi
```

Then install matching PyTorch:
- **CUDA 11.8**: `--index-url https://download.pytorch.org/whl/cu118`
- **CUDA 12.1**: `--index-url https://download.pytorch.org/whl/cu121`
- **CPU only**: `--index-url https://download.pytorch.org/whl/cpu`

---

## Next Steps After Setup

1. **Upload music files** (if not done yet):
   ```bash
   rsync -avz --partial --progress --timeout=300 \
     /Users/andreyvlasenko/output/ \
     g260141@levante.dkrz.de:/work/gg0302/g260141/Jingle/output/
   ```

2. **Generate training dataset**:
   ```bash
   source multipattern_env/bin/activate
   python create_dataset_multipattern.py
   ```

3. **Submit training job**:
   ```bash
   sbatch run_levante.sh
   ```

4. **Monitor progress**:
   ```bash
   tail -f logs/train-<job_id>.out
   ```
