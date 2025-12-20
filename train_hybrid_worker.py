#!/usr/bin/env python3
"""
Hybrid DDP Training Worker

Combines:
1. Baseline functionality (GAN, full metrics, all printouts)
2. Adaptive window selection (3-window pairs with temporal/tonal transforms)
3. All creative agent features (complementarity, correlation analysis)
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*git.*')
warnings.filterwarnings('ignore', message='.*No device id is provided.*')

import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import sys
import signal
from encodec import EncodecModel

try:
    import mlflow
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("Warning: mlflow not installed. Logging disabled.")

# Import models and agents
from adaptive_window_agent import AdaptiveWindowCreativeAgent
from audio_discriminator import AudioDiscriminator, discriminator_loss, generator_loss, roll_targets
from dataset_wav_pairs_24sec import AudioPairsDataset24sec, collate_fn
from training.losses import rms_loss, stft_loss, mel_loss, combined_loss, spectral_outlier_penalty


def setup_ddp(rank, world_size):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
        if rank == 0:
            print(f"WARNING: MASTER_PORT not set, using default 12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


def load_encodec_model(bandwidth=6.0, sample_rate=24000, device='cuda'):
    """Load and freeze EnCodec model"""
    if sample_rate == 24000:
        model = EncodecModel.encodec_model_24khz()
    else:
        raise ValueError(f"Unsupported sample rate: {sample_rate}")
    model.set_target_bandwidth(bandwidth)
    model = model.to(device)
    model.train()  # Keep in train mode but freeze parameters
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def encode_audio_batch(audio_batch, encodec_model, target_frames=1200):
    """
    Encode batch of audio with EnCodec.
    
    Args:
        audio_batch: [B, 1, samples] - mono audio (24 sec = 576,000 samples)
        encodec_model: EnCodec model
        target_frames: Target number of frames (1200 = 24 sec at 50 fps)
    
    Returns:
        encoded: [B, 128, target_frames]
    """
    with torch.no_grad():
        latents = encodec_model.encoder(audio_batch)  # [B, 128, T]
        
        # Resample to target frames if needed
        if latents.shape[-1] != target_frames:
            latents = torch.nn.functional.interpolate(
                latents,
                size=target_frames,
                mode='linear',
                align_corners=False
            )
    
    return latents


def _create_waveform_visualization(audio_tensor, label, width=80):
    """Create ASCII waveform visualization"""
    # audio_tensor: [samples] or [1, samples]
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze()
    
    # Downsample to width points
    step = max(1, len(audio_tensor) // width)
    samples = audio_tensor[::step][:width].cpu().numpy()
    
    # Normalize to [-1, 1]
    max_val = np.abs(samples).max()
    if max_val > 0:
        samples = samples / max_val
    
    # Create visualization
    viz = []
    for val in samples:
        if val > 0:
            bar_len = int(val * 20)
            viz.append('▁▂▃▄▅▆▇█'[min(bar_len // 3, 7)])
        else:
            bar_len = int(-val * 20)
            viz.append('▁▂▃▄▅▆▇█'[min(bar_len // 3, 7)])
    
    return f"{label}: {''.join(viz)}"


def train_epoch(model, dataloader, encodec_model, optimizer, rank, world_size, args, epoch,
                discriminator=None, disc_optimizer=None):
    """
    Train for one epoch with adaptive windows + GAN + full metrics.
    
    Returns:
        Comprehensive metrics dict with all training statistics
    """
    model.train()
    if discriminator is not None:
        discriminator.train()
    
    # Loss accumulators
    total_loss = 0.0
    total_rms_input = 0.0
    total_rms_target = 0.0
    total_spectral = 0.0
    total_mel = 0.0
    total_corr_penalty = 0.0
    total_novelty = 0.0  # Mask reg loss from creative agent
    total_balance_loss_raw = 0.0
    
    # GAN metrics
    total_gan_loss = 0.0
    total_disc_loss = 0.0
    total_disc_real_acc = 0.0
    total_disc_fake_acc = 0.0
    
    # Adaptive window statistics
    total_pair0_start = 0.0
    total_pair1_start = 0.0
    total_pair2_start = 0.0
    total_pair0_ratio = 0.0
    total_pair1_ratio = 0.0
    total_pair2_ratio = 0.0
    total_pair0_tonality = 0.0
    total_pair1_tonality = 0.0
    total_pair2_tonality = 0.0
    
    # Compositional agent statistics
    total_input_rhythm_w = 0.0
    total_input_harmony_w = 0.0
    total_target_rhythm_w = 0.0
    total_target_harmony_w = 0.0
    
    # Correlation analysis
    total_output_input_corr = 0.0
    total_output_target_corr = 0.0
    total_complementarity = 0.0
    total_mask_overlap = 0.0
    total_input_mask_mean = 0.0
    total_target_mask_mean = 0.0
    
    num_batches = 0
    total_batches = len(dataloader)
    
    for batch_idx, (audio_inputs, audio_targets) in enumerate(dataloader):
        audio_inputs = audio_inputs.to(rank)  # [B, 1, 576000]
        audio_targets = audio_targets.to(rank)
        
        # Unity test mode
        if args.unity_test:
            audio_targets = audio_inputs.clone()
            if rank == 0 and batch_idx == 0:
                print(f"\n🔍 Unity Test ENABLED: target = input (sanity check)")
        
        # Shuffle targets mode
        if args.shuffle_targets:
            indices = torch.randperm(audio_targets.size(0), device=audio_targets.device)
            audio_targets = audio_targets[indices]
            if rank == 0 and batch_idx == 0:
                print(f"🎲 Shuffle Targets ENABLED: random pairing for creativity")
        
        # Encode audio to latent space
        encoded_inputs = encode_audio_batch(audio_inputs, encodec_model)  # [B, 128, 1200]
        encoded_targets = encode_audio_batch(audio_targets, encodec_model)
        
        # Store original clean targets for loss computation
        original_targets = audio_targets.clone()
        
        # Forward pass through adaptive window agent
        # Returns: List of 3 outputs [B, 128, 800], List of 3 novelty losses, metadata
        outputs_list, novelty_losses, metadata = model(encoded_inputs, encoded_targets)
        
        # Average the 3 outputs (simple ensemble)
        # Each output is [B, 128, 800] (16 seconds compressed)
        # We need to upsample back to 1200 frames for consistent decoding
        
        # CRITICAL: Must upsample BEFORE averaging, not after
        # Use simple repeat + interpolate for robustness
        outputs_upsampled = []
        for output in outputs_list:
            B, D, T = output.shape
            # Method: Reshape and use adaptive_avg_pool1d for reliable upsampling
            # This is more explicit than interpolate
            output_reshaped = output.reshape(B * D, 1, T)  # [B*D, 1, 800]
            output_upsampled = torch.nn.functional.adaptive_avg_pool1d(
                output_reshaped, output_size=1200
            )  # [B*D, 1, 1200]
            output_1200 = output_upsampled.reshape(B, D, 1200)  # [B, 128, 1200]
            outputs_upsampled.append(output_1200)
        
        # Average the 3 upsampled outputs
        encoded_output = torch.stack(outputs_upsampled, dim=0).mean(dim=0)  # [B, 128, 1200]
        
        # Average novelty losses
        mean_novelty_loss = torch.stack(novelty_losses).mean()
        
        # Decode to audio space (with no_grad to prevent RNN backward error)
        with torch.no_grad():
            output_audio = encodec_model.decoder(encoded_output.detach())  # [B, 1, 576000]
            input_audio = encodec_model.decoder(encoded_inputs.detach())
        
        # Combined loss with all components
        loss, rms_input_val, rms_target_val, spec_val, mel_val, corr_penalty_val = combined_loss(
            output_audio, input_audio, original_targets,
            args.loss_weight_input, args.loss_weight_target,
            args.loss_weight_spectral, args.loss_weight_mel,
            weight_correlation=args.corr_weight
        )
        
        # Add novelty/mask regularization loss
        loss = loss + args.mask_reg_weight * mean_novelty_loss
        
        # Extract balance loss from metadata (if creative agent provides it)
        balance_loss_raw = 0.0
        if 'balance_loss' in metadata:
            balance_loss_raw = metadata['balance_loss']
            loss = loss + args.balance_loss_weight * balance_loss_raw
        
        # Spectral outlier penalty
        spectral_penalty = spectral_outlier_penalty(
            output_audio,
            sample_rate=24000,
            freq_bands=[(1800, 2200), (3800, 4200), (5800, 6200)]
        )
        loss = loss + 0.01 * spectral_penalty
        
        # Compute correlation analysis (output vs input/target)
        with torch.no_grad():
            output_flat = output_audio.reshape(-1)
            input_flat = input_audio.reshape(-1)
            target_flat = original_targets.reshape(-1)
            
            # Correlations
            output_input_corr = torch.corrcoef(torch.stack([output_flat, input_flat]))[0, 1]
            output_target_corr = torch.corrcoef(torch.stack([output_flat, target_flat]))[0, 1]
        
        # GAN training
        gan_g_loss = torch.tensor(0.0, device=rank)
        disc_loss_value = torch.tensor(0.0, device=rank)
        disc_real_acc = 0.0
        disc_fake_acc = 0.0
        
        if discriminator is not None and args.gan_weight > 0:
            # Train discriminator (every disc_update_freq batches)
            if batch_idx % args.disc_update_freq == 0:
                disc_optimizer.zero_grad()
                
                # Real samples: use encoded targets
                real_samples = roll_targets(encoded_targets.detach())
                real_logits = discriminator(real_samples)
                
                # Fake samples: use averaged output (detach from generator)
                fake_samples = encoded_output.detach()
                fake_logits = discriminator(fake_samples)
                
                # Discriminator loss
                disc_loss, disc_real_acc, disc_fake_acc = discriminator_loss(real_logits, fake_logits)
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                disc_optimizer.step()
                
                disc_loss_value = disc_loss.detach()
            
            # Generator adversarial loss
            fake_logits_for_gen = discriminator(encoded_output)
            gan_g_loss = generator_loss(fake_logits_for_gen)
            loss = loss + args.gan_weight * gan_g_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient debugging (first batch only)
        if batch_idx == 0 and epoch == 1 and rank == 0:
            print(f"\n{'='*80}")
            print(f"GRADIENT DEBUGGING (First batch, Epoch {epoch})")
            print(f"{'='*80}")
            total_grad_norm = 0.0
            grad_info = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm ** 2
                    grad_info.append((name, grad_norm, param.grad.abs().mean().item(), param.grad.abs().max().item()))
            
            total_grad_norm = total_grad_norm ** 0.5
            print(f"Total gradient norm (before clipping): {total_grad_norm:.6f}")
            
            print("\nTop 10 parameters by gradient norm:")
            for name, norm, mean, max_val in sorted(grad_info, key=lambda x: -x[1])[:10]:
                print(f"  {name:50s}: norm={norm:.6f}, mean={mean:.6f}, max={max_val:.6f}")
            
            zero_grad = [name for name, norm, _, _ in grad_info if norm < 1e-8]
            if zero_grad:
                print(f"\n⚠️  WARNING: {len(zero_grad)} parameters have near-zero gradients:")
                for name in zero_grad[:5]:
                    print(f"    {name}")
                if len(zero_grad) > 5:
                    print(f"    ... and {len(zero_grad)-5} more")
            
            print(f"{'='*80}\n")
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_rms_input += rms_input_val.item() if isinstance(rms_input_val, torch.Tensor) else rms_input_val
        total_rms_target += rms_target_val.item() if isinstance(rms_target_val, torch.Tensor) else rms_target_val
        total_spectral += spec_val.item() if isinstance(spec_val, torch.Tensor) else spec_val
        total_mel += mel_val.item() if isinstance(mel_val, torch.Tensor) else mel_val
        total_corr_penalty += corr_penalty_val.item() if isinstance(corr_penalty_val, torch.Tensor) else corr_penalty_val
        total_novelty += mean_novelty_loss.item()
        total_balance_loss_raw += balance_loss_raw if isinstance(balance_loss_raw, float) else balance_loss_raw.item()
        
        if discriminator is not None and args.gan_weight > 0:
            total_gan_loss += gan_g_loss.item()
            total_disc_loss += disc_loss_value.item()
            total_disc_real_acc += disc_real_acc
            total_disc_fake_acc += disc_fake_acc
        
        # Accumulate correlation metrics
        total_output_input_corr += output_input_corr.item()
        total_output_target_corr += output_target_corr.item()
        
        # Accumulate window statistics
        if 'pairs' in metadata and len(metadata['pairs']) >= 3:
            total_pair0_start += metadata['pairs'][0].get('start_input_mean', 0.0)
            total_pair1_start += metadata['pairs'][1].get('start_input_mean', 0.0)
            total_pair2_start += metadata['pairs'][2].get('start_input_mean', 0.0)
            total_pair0_ratio += metadata['pairs'][0].get('ratio_input_mean', 0.0)
            total_pair1_ratio += metadata['pairs'][1].get('ratio_input_mean', 0.0)
            total_pair2_ratio += metadata['pairs'][2].get('ratio_input_mean', 0.0)
            total_pair0_tonality += metadata['pairs'][0].get('tonality_input_mean', 0.0)
            total_pair1_tonality += metadata['pairs'][1].get('tonality_input_mean', 0.0)
            total_pair2_tonality += metadata['pairs'][2].get('tonality_input_mean', 0.0)
        
        # Accumulate compositional agent statistics
        if 'compositional_stats' in metadata:
            stats = metadata['compositional_stats']
            total_input_rhythm_w += stats.get('input_rhythm_weight', 0.0)
            total_input_harmony_w += stats.get('input_harmony_weight', 0.0)
            total_target_rhythm_w += stats.get('target_rhythm_weight', 0.0)
            total_target_harmony_w += stats.get('target_harmony_weight', 0.0)
        
        num_batches += 1
        
        # Progress update (rank 0 only, every 20 batches)
        if rank == 0 and (batch_idx % 20 == 0 or batch_idx == total_batches - 1):
            progress_pct = (batch_idx + 1) / total_batches * 100
            print(f"Epoch {epoch}: {batch_idx+1}/{total_batches} ({progress_pct:.0f}%) - "
                  f"loss={loss.item():.4f}, novelty={mean_novelty_loss.item():.4f}, "
                  f"rms_in={rms_input_val if isinstance(rms_input_val, float) else rms_input_val.item():.4f}, "
                  f"rms_tgt={rms_target_val if isinstance(rms_target_val, float) else rms_target_val.item():.4f}")
        
        # Debug: Print first batch window selection
        if rank == 0 and batch_idx == 0 and epoch == 1:
            print(f"\n🎯 First Batch Window Selection:")
            if 'pairs' in metadata:
                for i, pair in enumerate(metadata['pairs'][:3]):
                    print(f"  Pair {i}: start={pair.get('start_input_mean', 0):.1f}f, "
                          f"ratio={pair.get('ratio_input_mean', 0):.2f}x, "
                          f"tonality={pair.get('tonality_input_mean', 0):.2f}")
    
    # Synchronize metrics across GPUs
    avg_loss_tensor = torch.tensor([total_loss / num_batches], device=rank)
    avg_novelty_tensor = torch.tensor([total_novelty / num_batches], device=rank)
    avg_spectral_tensor = torch.tensor([total_spectral / num_batches], device=rank)
    avg_mel_tensor = torch.tensor([total_mel / num_batches], device=rank)
    avg_rms_input_tensor = torch.tensor([total_rms_input / num_batches], device=rank)
    avg_rms_target_tensor = torch.tensor([total_rms_target / num_batches], device=rank)
    avg_corr_penalty_tensor = torch.tensor([total_corr_penalty / num_batches], device=rank)
    
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_novelty_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_spectral_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_mel_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_rms_input_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_rms_target_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_corr_penalty_tensor, op=dist.ReduceOp.SUM)
    
    avg_loss = avg_loss_tensor.item() / world_size
    avg_novelty = avg_novelty_tensor.item() / world_size
    avg_spectral = avg_spectral_tensor.item() / world_size
    avg_mel = avg_mel_tensor.item() / world_size
    avg_rms_input = avg_rms_input_tensor.item() / world_size
    avg_rms_target = avg_rms_target_tensor.item() / world_size
    avg_corr_penalty = avg_corr_penalty_tensor.item() / world_size
    
    # Prepare return dict
    metrics = {
        'loss': avg_loss,
        'novelty': avg_novelty,
        'spectral': avg_spectral,
        'mel': avg_mel,
        'rms_input': avg_rms_input,
        'rms_target': avg_rms_target,
        'corr_penalty': avg_corr_penalty,
        'balance_loss_raw': total_balance_loss_raw / num_batches,
        'output_input_corr': total_output_input_corr / num_batches,
        'output_target_corr': total_output_target_corr / num_batches,
    }
    
    # Add GAN metrics if applicable
    if discriminator is not None and args.gan_weight > 0:
        metrics['gan_loss'] = total_gan_loss / num_batches
        metrics['disc_loss'] = total_disc_loss / max(1, num_batches // args.disc_update_freq)
        metrics['disc_real_acc'] = total_disc_real_acc / max(1, num_batches // args.disc_update_freq)
        metrics['disc_fake_acc'] = total_disc_fake_acc / max(1, num_batches // args.disc_update_freq)
    
    # Add window stats
    if num_batches > 0:
        metrics['window_stats'] = {
            'pair0_start': total_pair0_start / num_batches,
            'pair1_start': total_pair1_start / num_batches,
            'pair2_start': total_pair2_start / num_batches,
            'pair0_ratio': total_pair0_ratio / num_batches,
            'pair1_ratio': total_pair1_ratio / num_batches,
            'pair2_ratio': total_pair2_ratio / num_batches,
            'pair0_tonality': total_pair0_tonality / num_batches,
            'pair1_tonality': total_pair1_tonality / num_batches,
            'pair2_tonality': total_pair2_tonality / num_batches,
        }
        
        # Add compositional stats
        metrics['component_stats'] = {
            'input_rhythm': total_input_rhythm_w / num_batches,
            'input_harmony': total_input_harmony_w / num_batches,
            'target_rhythm': total_target_rhythm_w / num_batches,
            'target_harmony': total_target_harmony_w / num_batches,
        }
    
    return metrics


def validate(model, dataloader, encodec_model, rank, world_size, args):
    """Validation loop with full metrics"""
    model.eval()
    
    total_loss = 0.0
    total_novelty = 0.0
    total_spectral = 0.0
    total_mel = 0.0
    total_rms_input = 0.0
    total_rms_target = 0.0
    num_batches = 0
    total_batches = len(dataloader)
    
    # Store first sample for visualization
    first_input_audio = None
    first_target_audio = None
    first_output_audio = None
    
    with torch.no_grad():
        for batch_idx, (audio_inputs, audio_targets) in enumerate(dataloader):
            audio_inputs = audio_inputs.to(rank)
            audio_targets = audio_targets.to(rank)
            
            # Encode
            encoded_inputs = encode_audio_batch(audio_inputs, encodec_model)
            encoded_targets = encode_audio_batch(audio_targets, encodec_model)
            
            # Forward
            outputs_list, novelty_losses, metadata = model(encoded_inputs, encoded_targets)
            
            # Average outputs with upsampling
            outputs_upsampled = []
            for output in outputs_list:
                B, D, T = output.shape
                # Use adaptive pooling for robust upsampling
                output_reshaped = output.reshape(B * D, 1, T)
                output_upsampled = torch.nn.functional.adaptive_avg_pool1d(
                    output_reshaped, output_size=1200
                )
                output_1200 = output_upsampled.reshape(B, D, 1200)
                outputs_upsampled.append(output_1200)
            encoded_output = torch.stack(outputs_upsampled, dim=0).mean(dim=0)
            mean_novelty_loss = torch.stack(novelty_losses).mean()
            
            # Decode
            output_audio = encodec_model.decoder(encoded_output.detach())
            input_audio = encodec_model.decoder(encoded_inputs.detach())
            
            # Loss
            loss, rms_input_val, rms_target_val, spec_val, mel_val, corr_val = combined_loss(
                output_audio, input_audio, audio_targets,
                args.loss_weight_input, args.loss_weight_target,
                args.loss_weight_spectral, args.loss_weight_mel,
                weight_correlation=0.0
            )
            
            loss = loss + args.mask_reg_weight * mean_novelty_loss
            
            # Store first sample
            if batch_idx == 0:
                first_input_audio = input_audio[0].cpu()
                first_target_audio = audio_targets[0].cpu()
                first_output_audio = output_audio[0].cpu()
            
            # Accumulate
            total_loss += loss.item()
            total_novelty += mean_novelty_loss.item()
            total_spectral += spec_val.item() if isinstance(spec_val, torch.Tensor) else spec_val
            total_mel += mel_val.item() if isinstance(mel_val, torch.Tensor) else mel_val
            total_rms_input += rms_input_val.item() if isinstance(rms_input_val, torch.Tensor) else rms_input_val
            total_rms_target += rms_target_val.item() if isinstance(rms_target_val, torch.Tensor) else rms_target_val
            num_batches += 1
            
            # Progress (rank 0 only, every 10 batches)
            if rank == 0 and (batch_idx % 10 == 0 or batch_idx == total_batches - 1):
                progress_pct = (batch_idx + 1) / total_batches * 100
                print(f"Validation: {batch_idx+1}/{total_batches} ({progress_pct:.0f}%) - "
                      f"loss={loss.item():.4f}, "
                      f"rms_in={rms_input_val if isinstance(rms_input_val, float) else rms_input_val.item():.4f}, "
                      f"rms_tgt={rms_target_val if isinstance(rms_target_val, float) else rms_target_val.item():.4f}")
    
    # Display waveform visualization (rank 0 only)
    if rank == 0 and first_input_audio is not None:
        print()  # New line after progress
        
        input_rms = torch.sqrt(torch.mean(first_input_audio ** 2))
        target_rms = torch.sqrt(torch.mean(first_target_audio ** 2))
        output_rms = torch.sqrt(torch.mean(first_output_audio ** 2))
        
        print(f"\n  🎵 Waveform Visualization (First validation sample):")
        print(f"  {_create_waveform_visualization(first_input_audio, 'Input ')}")
        print(f"  {_create_waveform_visualization(first_target_audio, 'Target')}")
        print(f"  {_create_waveform_visualization(first_output_audio, 'Output')}")
        
        rms_ratio = output_rms.item() / input_rms.item() if input_rms.item() > 1e-8 else 0.0
        print(f"\n  📊 Audio Statistics:")
        print(f"     RMS: Input={input_rms.item():.4f}  Target={target_rms.item():.4f}  Output={output_rms.item():.4f}")
        print(f"     Output/Input ratio: {rms_ratio:.2f}x")
    
    # Synchronize metrics
    avg_loss_tensor = torch.tensor([total_loss / num_batches], device=rank)
    avg_novelty_tensor = torch.tensor([total_novelty / num_batches], device=rank)
    avg_spectral_tensor = torch.tensor([total_spectral / num_batches], device=rank)
    avg_mel_tensor = torch.tensor([total_mel / num_batches], device=rank)
    avg_rms_input_tensor = torch.tensor([total_rms_input / num_batches], device=rank)
    avg_rms_target_tensor = torch.tensor([total_rms_target / num_batches], device=rank)
    
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_novelty_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_spectral_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_mel_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_rms_input_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_rms_target_tensor, op=dist.ReduceOp.SUM)
    
    return {
        'loss': avg_loss_tensor.item() / world_size,
        'novelty': avg_novelty_tensor.item() / world_size,
        'spectral': avg_spectral_tensor.item() / world_size,
        'mel': avg_mel_tensor.item() / world_size,
        'rms_input': avg_rms_input_tensor.item() / world_size,
        'rms_target': avg_rms_target_tensor.item() / world_size,
    }


def worker_main(rank, world_size, args):
    """Main worker function"""
    print(f"[Rank {rank}] Starting worker...")
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    import numpy as np
    np.random.seed(args.seed + rank)
    
    # Setup
    setup_ddp(rank, world_size)
    torch.cuda.set_device(rank)
    
    # Load EnCodec
    encodec_model = load_encodec_model(
        bandwidth=args.encodec_bandwidth,
        sample_rate=args.encodec_sr,
        device=rank
    )
    
    # Create adaptive window agent model
    if rank == 0:
        print("="*80)
        print("🎯 HYBRID TRAINING: Adaptive Windows + GAN + Full Metrics")
        print("="*80)
        print(f"  AdaptiveWindowCreativeAgent:")
        print(f"    - 3 window pairs with selection")
        print(f"    - Temporal compression (1.0-1.5x)")
        print(f"    - Tonality reduction")
        print(f"    - Compositional agent (rhythm/harmony)")
        print("="*80)
    
    model = AdaptiveWindowCreativeAgent(
        encoding_dim=args.encoding_dim,
        num_pairs=args.num_pairs
    ).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if rank == 0:
        print(f"  Model Parameters:")
        print(f"    Total: {total_params:,}")
        print(f"    Trainable: {trainable_params:,}")
        print("="*80)
    
    # Create discriminator if GAN training enabled
    discriminator = None
    disc_optimizer = None
    if args.gan_weight > 0:
        discriminator = AudioDiscriminator(encoding_dim=args.encoding_dim).to(rank)
        discriminator = DDP(discriminator, device_ids=[rank])
        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=args.disc_lr,
            weight_decay=args.weight_decay
        )
        
        disc_params = sum(p.numel() for p in discriminator.parameters())
        if rank == 0:
            print(f"  🎭 GAN Discriminator:")
            print(f"    Parameters: {disc_params:,}")
            print(f"    GAN weight: {args.gan_weight}")
            print(f"    Disc update freq: {args.disc_update_freq}")
            print("="*80)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Dataset
    if rank == 0:
        print(f"Loading dataset from {args.dataset_dir}...")
    
    train_dataset = AudioPairsDataset24sec(
        data_folder=args.dataset_dir,
        split='train',
        target_sr=args.encodec_sr
    )
    val_dataset = AudioPairsDataset24sec(
        data_folder=args.dataset_dir,
        split='val',
        target_sr=args.encodec_sr
    )
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    if rank == 0:
        print(f"  Train: {len(train_dataset)} pairs ({len(train_loader)} batches)")
        print(f"  Val: {len(val_dataset)} pairs ({len(val_loader)} batches)")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Total batch size: {args.batch_size * world_size}")
        print("="*80)
    
    # Training loop
    start_epoch = 1
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, encodec_model, optimizer,
            rank, world_size, args, epoch,
            discriminator=discriminator, disc_optimizer=disc_optimizer
        )
        
        # Validate
        val_metrics = validate(model, val_loader, encodec_model, rank, world_size, args)
        
        # Print comprehensive results (rank 0 only)
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{args.epochs} Results:")
            print(f"{'='*80}")
            
            # Training metrics
            print(f"  📊 Train Metrics:")
            print(f"     Total Loss:    {train_metrics['loss']:.4f}")
            print(f"     RMS Input:     {train_metrics['rms_input']:.4f} (weight: {args.loss_weight_input})")
            print(f"     RMS Target:    {train_metrics['rms_target']:.4f} (weight: {args.loss_weight_target})")
            print(f"     Novelty Loss:  {train_metrics['novelty']:.4f} (weight: {args.mask_reg_weight})")
            if args.loss_weight_spectral > 0:
                print(f"     Spectral Loss: {train_metrics['spectral']:.4f} (weight: {args.loss_weight_spectral})")
            if args.loss_weight_mel > 0:
                print(f"     Mel Loss:      {train_metrics['mel']:.4f} (weight: {args.loss_weight_mel})")
            if args.corr_weight > 0:
                print(f"     Correlation Penalty: {train_metrics['corr_penalty']:.4f} (weight: {args.corr_weight})")
            if 'balance_loss_raw' in train_metrics and train_metrics['balance_loss_raw'] > 0:
                print(f"     Balance Loss:  {train_metrics['balance_loss_raw']:.4f} (×{args.balance_loss_weight} weight = {train_metrics['balance_loss_raw']*args.balance_loss_weight:.4f})")
            
            # GAN metrics
            if discriminator is not None and args.gan_weight > 0:
                print(f"\n  🎭 GAN Training:")
                print(f"     Gen Loss:      {train_metrics.get('gan_loss', 0):.4f}")
                print(f"     Disc Loss:     {train_metrics.get('disc_loss', 0):.4f}")
                print(f"     Disc Acc:      Real={train_metrics.get('disc_real_acc', 0):.1%} / Fake={train_metrics.get('disc_fake_acc', 0):.1%}")
            
            # Correlation analysis
            print(f"\n  📊 Output Correlation Analysis:")
            print(f"     Output→Input:  {train_metrics['output_input_corr']:.3f}")
            print(f"     Output→Target: {train_metrics['output_target_corr']:.3f}")
            if abs(train_metrics['output_target_corr']) > abs(train_metrics['output_input_corr']) * 2:
                print(f"     ⚠️  OUTPUT IS COPYING TARGET! (target corr >> input corr)")
            elif abs(train_metrics['output_input_corr']) > abs(train_metrics['output_target_corr']) * 2:
                print(f"     ⚠️  OUTPUT IS COPYING INPUT! (input corr >> target corr)")
            else:
                print(f"     ✓ Output appears to mix both sources")
            
            # Compositional agent stats
            if 'component_stats' in train_metrics:
                stats = train_metrics['component_stats']
                print(f"\n  🎼 Compositional Agent (Component Weights):")
                print(f"     Input:  rhythm={stats['input_rhythm']:.3f}, harmony={stats['input_harmony']:.3f}")
                print(f"     Target: rhythm={stats['target_rhythm']:.3f}, harmony={stats['target_harmony']:.3f}")
            
            # Window selection stats
            if 'window_stats' in train_metrics:
                ws = train_metrics['window_stats']
                print(f"\n  🎯 Adaptive Window Selection:")
                print(f"     Pair 0: start={ws['pair0_start']:6.1f}f  ratio={ws['pair0_ratio']:.2f}x  tonality={ws['pair0_tonality']:.2f}")
                print(f"     Pair 1: start={ws['pair1_start']:6.1f}f  ratio={ws['pair1_ratio']:.2f}x  tonality={ws['pair1_tonality']:.2f}")
                print(f"     Pair 2: start={ws['pair2_start']:6.1f}f  ratio={ws['pair2_ratio']:.2f}x  tonality={ws['pair2_tonality']:.2f}")
                
                start_std = torch.std(torch.tensor([
                    ws['pair0_start'], ws['pair1_start'], ws['pair2_start']
                ])).item()
                ratio_std = torch.std(torch.tensor([
                    ws['pair0_ratio'], ws['pair1_ratio'], ws['pair2_ratio']
                ])).item()
                
                print(f"     Diversity: start_std={start_std:.1f}f, ratio_std={ratio_std:.3f}x")
                if start_std < 10:
                    print(f"     ⚠️  WARNING: Low window diversity")
                elif start_std > 50:
                    print(f"     ✓ Good window diversity")
            
            # Validation metrics
            print(f"\n  📈 Validation Metrics:")
            print(f"     Total Loss:    {val_metrics['loss']:.4f}")
            print(f"     RMS Input:     {val_metrics['rms_input']:.4f}")
            print(f"     RMS Target:    {val_metrics['rms_target']:.4f}")
            print(f"     Novelty Loss:  {val_metrics['novelty']:.4f}")
            if args.loss_weight_spectral > 0:
                print(f"     Spectral Loss: {val_metrics['spectral']:.4f}")
            if args.loss_weight_mel > 0:
                print(f"     Mel Loss:      {val_metrics['mel']:.4f}")
            
            print(f"\n  🔧 Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            print("="*80)
        
        # Save checkpoint
        if rank == 0 and epoch % args.save_every == 0:
            checkpoint_path = f"{args.checkpoint_dir}/hybrid_epoch_{epoch}.pt"
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'args': vars(args),
            }
            if discriminator is not None:
                checkpoint_data['discriminator_state_dict'] = discriminator.module.state_dict()
                checkpoint_data['disc_optimizer_state_dict'] = disc_optimizer.state_dict()
            torch.save(checkpoint_data, checkpoint_path)
            print(f"\n💾 Checkpoint saved: {checkpoint_path}\n")
    
    cleanup_ddp()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid Training: Adaptive Windows + GAN')
    
    # Training params
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_hybrid')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    
    # Architecture params
    parser.add_argument('--num_pairs', type=int, default=3)
    parser.add_argument('--encoding_dim', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_transformer_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # EnCodec params
    parser.add_argument('--encodec_bandwidth', type=float, default=6.0)
    parser.add_argument('--encodec_sr', type=int, default=24000)
    
    # Loss weights
    parser.add_argument('--loss_weight_input', type=float, default=0.3)
    parser.add_argument('--loss_weight_target', type=float, default=0.3)
    parser.add_argument('--loss_weight_spectral', type=float, default=0.01)
    parser.add_argument('--loss_weight_mel', type=float, default=0.01)
    parser.add_argument('--corr_weight', type=float, default=0.5)
    parser.add_argument('--anti_cheating', type=float, default=0.2)
    parser.add_argument('--mask_reg_weight', type=float, default=0.1)
    parser.add_argument('--balance_loss_weight', type=float, default=15.0)
    
    # GAN params
    parser.add_argument('--gan_weight', type=float, default=0.1)
    parser.add_argument('--disc_lr', type=float, default=5e-5)
    parser.add_argument('--disc_update_freq', type=int, default=1)
    
    # Agent params
    parser.add_argument('--use_compositional_agent', type=str, default='true')
    
    # Special modes
    parser.add_argument('--unity_test', action='store_true')
    parser.add_argument('--shuffle_targets', action='store_true')
    
    # DDP params
    parser.add_argument('--world_size', type=int, default=4)
    
    args = parser.parse_args()
    
    # Launch workers
    torch.multiprocessing.spawn(
        worker_main,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )
