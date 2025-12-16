# Files Manifest - Jingle_D

Clean repository with only essential training, inference, and documentation files.

## Core Model Files (5 files)

| File | Lines | Purpose |
|------|-------|---------|
| `model_simple_transformer.py` | ~800 | 2-stage cascade transformer architecture |
| `creative_agent.py` | ~200 | Attention-based complementary mask generator |
| `audio_discriminator.py` | ~150 | GAN discriminator for adversarial training |
| `correlation_penalty.py` | ~50 | Anti-modulation loss function |
| `complementary_masking.py` | ~100 | Legacy fixed masking (reference) |

## Training System (3 files)

| File | Lines | Purpose |
|------|-------|---------|
| `train_simple_worker.py` | ~1100 | DDP training worker with full training loop |
| `train_simple_ddp.py` | ~150 | Multi-GPU launcher using torch.multiprocessing |
| `training/losses.py` | ~200 | Loss functions (RMS, spectral, mel) |

## Launch Scripts (4 files)

| File | Purpose |
|------|---------|
| `run_train_creative_agent_fixed.sh` | Main training script (epochs 1-200) |
| `run_train_creative_agent_resume.sh` | Resume from checkpoint (epoch 20+) |
| `run_train_creative_agent_push_complementarity.sh` | **NEW** - Strong complementarity (mask_reg_weight=0.5) |
| `submit_creative_agent.sh` | SLURM submission wrapper |

## Inference (2 files)

| File | Lines | Purpose |
|------|-------|---------|
| `inference_cascade.py` | ~250 | Generate audio from trained model |
| `inference_sample.py` | ~200 | Single sample inference with analysis |

## Dataset Tools (6 files)

| File | Purpose |
|------|---------|
| `dataset_wav_pairs.py` | PyTorch Dataset for paired WAV loading |
| `create_dataset_pairs_wav.py` | Create training dataset from audio files |
| `dataset_encoded_expanded.py` | EnCodec pre-encoded dataset (optional) |
| `extract_rhythm.py` | Extract rhythmic components |
| `extract_melody.py` | Extract melodic components |
| `extract_harmony.py` | Extract harmonic components |

## Utilities (4 files)

| File | Purpose |
|------|---------|
| `sync_to_remote.sh` | Sync code to SLURM cluster (levante.dkrz.de) |
| `verify_sync.sh` | Verify remote sync status |
| `view_mlflow_data.py` | View training metrics without MLflow UI |
| `check_gpus.sh` | GPU status check |

## Testing (8 files)

| File | Purpose |
|------|---------|
| `test_simple_setup.py` | Basic model instantiation test |
| `test_masking.py` | Creative agent masking tests |
| `test_spectral_losses.py` | Loss function validation |
| `test_model_learning.py` | Gradient flow verification |
| `test_creative_integration.py` | End-to-end creative agent test |
| `test_full_pipeline.py` | Complete training pipeline test |
| `test_pipeline_dry_run.py` | Dry run without GPU |
| `test_random_inference.py` | Random input inference test |

## Documentation (21 files)

### Core Documentation (4 files)

| File | Description |
|------|-------------|
| `README.md` | **Main project documentation** |
| `SESSION_CHECKPOINT.md` | Detailed training history and current status |
| `GRADIENT_EXPLOSION_FIX.md` | Architecture debugging notes (3-stage → 2-stage) |
| `README_CREATIVE_AGENT.md` | Creative agent design and implementation |

### Architecture Documentation (6 files)

| File | Focus |
|------|-------|
| `ARCHITECTURE_2STAGE_DETAILED.md` | 2-stage cascade architecture details |
| `ARCHITECTURE_CHANGE_2STAGE.md` | Migration from 3-stage to 2-stage |
| `GRADIENT_EXPLOSION_ARCHITECTURE.md` | Gradient stability analysis |
| `SKIP_CONNECTIONS_UPDATE.md` | Skip connection modifications |
| `CHANGES_RMS_FIX.md` | RMS loss computation fixes |
| `BALANCE_LOSS_DIAGNOSIS.md` | Mask balance implementation |

### Training Documentation (5 files)

| File | Focus |
|------|-------|
| `QUICKSTART_CREATIVE_AGENT.md` | Quick start guide |
| `QUICKSTART_RESUME.md` | Resume training guide |
| `README_LEVANTE.md` | SLURM cluster setup |
| `README_MASKING.md` | Masking strategy explanation |
| `README_SIMPLE.md` | Simple training walkthrough |

### Technical Notes (6 files)

| File | Focus |
|------|-------|
| `GRADIENT_FIX.md` | Gradient clipping and normalization |
| `NUMERICAL_DEBUGGING.md` | Numerical stability debugging |
| `QUADRATIC_PENALTY_FIX.md` | Complementarity loss optimization |
| `ANTI_MODULATION_COST.md` | Correlation penalty design |
| `SPECTRAL_LOSSES.md` | Spectral loss functions |
| `README_COMPOSITIONAL.md` | Alternative compositional agent |

## Alternative Implementations (2 files)

| File | Purpose |
|------|---------|
| `compositional_creative_agent.py` | Alternative compositional masking approach |
| `debug_validation.py` | Validation debugging utilities |

## SLURM Scripts (4 files)

| File | Purpose |
|------|---------|
| `create_dataset_levante.sh` | Dataset creation on cluster |
| `create_dataset_pairs_levante.sh` | Paired dataset on cluster |
| `create_dataset_encoded_levante.sh` | Pre-encoded dataset on cluster |
| `submit_creative_agent.sh` | SLURM job submission |

## Total Statistics

- **Total Files**: 62 essential files
- **Total Lines**: ~20,000 lines of code
- **Python Files**: 25 files (~5,000 LOC)
- **Shell Scripts**: 9 files (~1,000 LOC)
- **Documentation**: 21 markdown files (~10,000 LOC)
- **Tests**: 8 test files (~1,000 LOC)

## Excluded Files (via .gitignore)

### Large Data (not in repo)
- Checkpoints: `checkpoints/`, `best_checkpoints/` (~5GB)
- Dataset: `dataset/`, `dataset_pairs_wav/` (~50GB)
- Training outputs: `inference_outputs/`, `mlruns/` (~1GB)

### Deprecated Code (ignored)
- Old scripts: `model_cascade.py`, `pretrain_transformer.py`
- Unused training scripts: `run_train_alloc.sh`, `run_train_direct.sh`
- Old SLURM scripts: `submit_levante.sh`, `upload_to_levante.sh`

### Backup Directories (ignored)
- `bkup/`, `bkup_*/`, `14122025_2356/`
- `__pycache__/`, `.venv/`

## Key Files for GitHub Submission

### Essential (must include):
1. `README.md` - Project overview
2. `model_simple_transformer.py` - Core architecture
3. `creative_agent.py` - Mask generator
4. `train_simple_worker.py` - Training loop
5. `train_simple_ddp.py` - Multi-GPU launcher
6. `inference_cascade.py` - Audio generation
7. `run_train_creative_agent_fixed.sh` - Main training script

### Important (highly recommended):
8. `SESSION_CHECKPOINT.md` - Training history
9. `GRADIENT_EXPLOSION_FIX.md` - Architecture notes
10. `audio_discriminator.py` - GAN component
11. `correlation_penalty.py` - Loss function
12. `dataset_wav_pairs.py` - Dataset loader
13. All test files - Validation suite

### Supporting (complete picture):
14. All documentation markdown files
15. Dataset creation scripts
16. Utility scripts (sync, verify, view)
17. SLURM submission scripts
