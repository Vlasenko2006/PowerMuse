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
from training.debug_utils import (
    debug_gradients, 
    print_training_progress, 
    print_window_selection_debug,
    print_epoch_summary,
    MetricsAccumulator
)


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
    
    # Use MetricsAccumulator instead of manual tracking
    metrics = MetricsAccumulator()
    
    # Additional metrics not in accumulator (for compatibility)
    total_complementarity = 0.0
    total_mask_overlap = 0.0
    total_input_mask_mean = 0.0
    total_target_mask_mean = 0.0
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
        
        # The agent creates 3 pairs, each processed independently
        # For each pair: decode output, extract matching 16-sec segments, compute loss
        # Then average all losses (as in the backup version)
        
        pair_losses = []
        pair_rms_input = []
        pair_rms_target = []
        pair_spectral = []
        pair_mel = []
        pair_corr_penalty = []
        
        for idx, encoded_output in enumerate(outputs_list):
            # Decode this pair's output [B, 128, 800] → [B, 1, 256000] (10.67 sec)
            with torch.no_grad():
                output_audio = encodec_model.decoder(encoded_output.detach())
            
            # Extract center 10.67 seconds from input and target to match output length
            # 800 frames * 320 samples/frame = 256000 samples
            # Center offset: (576000 - 256000) / 2 = 160000
            input_10sec = audio_inputs[:, :, 160000:416000]  # [B, 1, 256000]
            target_10sec = original_targets[:, :, 160000:416000]  # [B, 1, 256000]
            
            # Compute loss for this pair
            pair_loss, rms_in, rms_tgt, spec, mel, corr = combined_loss(
                output_audio, input_10sec, target_10sec,
                args.loss_weight_input, args.loss_weight_target,
                args.loss_weight_spectral, args.loss_weight_mel,
                weight_correlation=args.corr_weight
            )
            
            pair_losses.append(pair_loss)
            pair_rms_input.append(rms_in)
            pair_rms_target.append(rms_tgt)
            pair_spectral.append(spec)
            pair_mel.append(mel)
            pair_corr_penalty.append(corr)
        
        # Average losses across all 3 pairs
        loss = torch.stack(pair_losses).mean()
        rms_input_val = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in pair_rms_input]).mean()
        rms_target_val = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in pair_rms_target]).mean()
        spec_val = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in pair_spectral]).mean()
        mel_val = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in pair_mel]).mean()
        corr_penalty_val = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in pair_corr_penalty]).mean()
        
        # Average novelty losses
        mean_novelty_loss = torch.stack(novelty_losses).mean()
        
        # Add novelty/mask regularization loss
        loss = loss + args.mask_reg_weight * mean_novelty_loss
        
        # ============ NEW: Ratio Supervision ============
        # 1. Ratio diversity loss - encourage different ratios across pairs
        if 'window_params' in metadata:
            all_ratios_input = []
            all_ratios_target = []
            for params in metadata['window_params']:
                all_ratios_input.append(params['ratio_input'])
                all_ratios_target.append(params['ratio_target'])
            
            # Stack all ratios: [num_pairs, B]
            ratios_input_stacked = torch.stack(all_ratios_input, dim=0)  # [3, B]
            ratios_target_stacked = torch.stack(all_ratios_target, dim=0)  # [3, B]
            
            # Encourage variance across pairs (averaged over batch)
            ratio_variance_input = torch.var(ratios_input_stacked, dim=0).mean()  # Variance across pairs
            ratio_variance_target = torch.var(ratios_target_stacked, dim=0).mean()
            
            # Negative variance loss = encourage higher variance
            ratio_diversity_loss = -(ratio_variance_input + ratio_variance_target)
            loss = loss + 0.1 * ratio_diversity_loss  # Weight: 0.1
        else:
            ratio_diversity_loss = torch.tensor(0.0, device=rank)
        
        # 2. Reconstruction loss - compare output to ORIGINAL uncompressed windows
        reconstruction_losses = []
        for idx, (encoded_output, params) in enumerate(zip(outputs_list, metadata['window_params'])):
            # Decode output
            with torch.no_grad():
                output_audio_decoded = encodec_model.decoder(encoded_output.detach())  # [B, 1, 256000]
            
            # Extract ORIGINAL windows (before compression) from 24-sec audio
            # Use the same start positions predicted by WindowSelector
            original_input_windows = []
            original_target_windows = []
            
            for b in range(audio_inputs.shape[0]):
                start_in = int(params['start_input'][b].item())
                start_tgt = int(params['start_target'][b].item())
                
                # Extract 800 frames from ENCODED (will decode to 256k samples)
                # This is the ORIGINAL window before compression
                start_in = max(0, min(start_in, encoded_inputs.shape[2] - 800))
                start_tgt = max(0, min(start_tgt, encoded_targets.shape[2] - 800))
                
                orig_in = encoded_inputs[b:b+1, :, start_in:start_in+800]  # [1, 128, 800]
                orig_tgt = encoded_targets[b:b+1, :, start_tgt:start_tgt+800]  # [1, 128, 800]
                
                original_input_windows.append(orig_in)
                original_target_windows.append(orig_tgt)
            
            # Decode original windows
            orig_in_batch = torch.cat(original_input_windows, dim=0)  # [B, 128, 800]
            orig_tgt_batch = torch.cat(original_target_windows, dim=0)  # [B, 128, 800]
            
            with torch.no_grad():
                orig_in_audio = encodec_model.decoder(orig_in_batch.detach())  # [B, 1, 256000]
                orig_tgt_audio = encodec_model.decoder(orig_tgt_batch.detach())  # [B, 1, 256000]
            
            # Compute reconstruction loss (should be close to original)
            recon_loss_input = F.mse_loss(output_audio_decoded, orig_in_audio)
            recon_loss_target = F.mse_loss(output_audio_decoded, orig_tgt_audio)
            
            # Average input and target reconstruction
            recon_loss = (recon_loss_input + recon_loss_target) / 2
            reconstruction_losses.append(recon_loss)
        
        # Average reconstruction loss across pairs
        mean_reconstruction_loss = torch.stack(reconstruction_losses).mean()
        loss = loss + 0.05 * mean_reconstruction_loss  # Weight: 0.05
        # ============ END: Ratio Supervision ============
        
        # Average the 3 encoded outputs for GAN training and other downstream uses
        encoded_output_avg = torch.stack(outputs_list, dim=0).mean(dim=0)  # [B, 128, 800]
        
        # Extract balance loss from metadata (if creative agent provides it)
        balance_loss_raw = 0.0
        if 'balance_loss' in metadata:
            balance_loss_raw = metadata['balance_loss']
            loss = loss + args.balance_loss_weight * balance_loss_raw
        
        # Decode first output for spectral penalty, correlation analysis, and GAN training
        with torch.no_grad():
            output_audio_first = encodec_model.decoder(outputs_list[0].detach())  # [B, 1, 256000]
        input_10sec = audio_inputs[:, :, 160000:416000]
        target_10sec = original_targets[:, :, 160000:416000]
        
        # Spectral outlier penalty
        spectral_penalty = spectral_outlier_penalty(
            output_audio_first,
            sample_rate=24000,
            freq_bands=[(1800, 2200), (3800, 4200), (5800, 6200)]
        )
        loss = loss + 0.01 * spectral_penalty
        
        # Compute correlation analysis (output vs input/target)
        with torch.no_grad():
            output_flat = output_audio_first.reshape(-1)
            input_flat = input_10sec.reshape(-1)
            target_flat = target_10sec.reshape(-1)
            
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
                fake_samples = encoded_output_avg.detach()
                fake_logits = discriminator(fake_samples)
                
                # Discriminator loss
                disc_loss, disc_real_acc, disc_fake_acc = discriminator_loss(real_logits, fake_logits)
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                disc_optimizer.step()
                
                disc_loss_value = disc_loss.detach()
            
            # Generator adversarial loss
            fake_logits_for_gen = discriminator(encoded_output_avg)
            gan_g_loss = generator_loss(fake_logits_for_gen)
            loss = loss + args.gan_weight * gan_g_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient debugging (first batch only)
        if batch_idx == 0 and epoch == 1:
            debug_gradients(model, epoch, rank)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics accumulator
        metrics.update(
            loss=loss,
            rms_input=rms_input_val,
            rms_target=rms_target_val,
            spectral=spec_val,
            mel=mel_val,
            ratio_diversity=ratio_diversity_loss,
            reconstruction=mean_reconstruction_loss,
            corr_penalty=corr_penalty_val,
            novelty=mean_novelty_loss,
            balance_loss_raw=balance_loss_raw,
            gan_loss=gan_g_loss,
            disc_loss=disc_loss_value,
            disc_real_acc=disc_real_acc,
            disc_fake_acc=disc_fake_acc,
            output_input_corr=output_input_corr,
            output_target_corr=output_target_corr,
            metadata=metadata
        )
        
        # Progress update
        print_training_progress(
            epoch, batch_idx, total_batches,
            loss, mean_novelty_loss, rms_input_val, rms_target_val, rank
        )
        
        # Debug: Print first batch window selection
        if batch_idx == 0:
            print_window_selection_debug(metadata, epoch, rank)
    
    # Get averaged metrics from accumulator
    avg_metrics = metrics.get_averages()
    window_stats = metrics.get_window_stats()
    
    # Synchronize core metrics across GPUs
    sync_tensors = {
        'loss': torch.tensor([avg_metrics['loss']], device=rank),
        'novelty': torch.tensor([avg_metrics['novelty']], device=rank),
        'spectral': torch.tensor([avg_metrics['spectral']], device=rank),
        'mel': torch.tensor([avg_metrics['mel']], device=rank),
        'rms_input': torch.tensor([avg_metrics['rms_input']], device=rank),
        'rms_target': torch.tensor([avg_metrics['rms_target']], device=rank),
        'corr_penalty': torch.tensor([avg_metrics['corr_penalty']], device=rank),
    }
    
    # All-reduce across GPUs
    for tensor in sync_tensors.values():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # Average across world size
    final_metrics = {
        key: tensor.item() / world_size 
        for key, tensor in sync_tensors.items()
    }
    
    # Add other metrics (already averaged locally)
    final_metrics.update({
        'balance_loss_raw': avg_metrics['balance_loss'],
        'output_input_corr': avg_metrics['output_input_corr'],
        'output_target_corr': avg_metrics['output_target_corr'],
    })
    
    # Add GAN metrics if applicable
    if discriminator is not None and args.gan_weight > 0:
        final_metrics['gan_loss'] = avg_metrics['gan_loss']
        final_metrics['disc_loss'] = avg_metrics['disc_loss']
        final_metrics['disc_real_acc'] = avg_metrics['disc_real_acc']
        final_metrics['disc_fake_acc'] = avg_metrics['disc_fake_acc']
    
    # Add component weights
    final_metrics['input_rhythm_w'] = avg_metrics['input_rhythm_w']
    final_metrics['input_harmony_w'] = avg_metrics['input_harmony_w']
    final_metrics['target_rhythm_w'] = avg_metrics['target_rhythm_w']
    final_metrics['target_harmony_w'] = avg_metrics['target_harmony_w']
    
    # Print epoch summary
    print_epoch_summary(epoch, final_metrics, window_stats, rank)
    
    return final_metrics


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
            
            # Compute loss for each of the 3 pairs and average
            pair_losses = []
            pair_rms_input = []
            pair_rms_target = []
            pair_spectral = []
            pair_mel = []
            
            for idx, encoded_output in enumerate(outputs_list):
                # Decode this pair's output
                output_audio = encodec_model.decoder(encoded_output.detach())  # [B, 1, 256000]
                
                # Extract center 10.67 seconds from input and target
                input_10sec = audio_inputs[:, :, 160000:416000]
                target_10sec = audio_targets[:, :, 160000:416000]
                
                # Compute loss for this pair
                pair_loss, rms_in, rms_tgt, spec, mel, corr = combined_loss(
                    output_audio, input_10sec, target_10sec,
                    args.loss_weight_input, args.loss_weight_target,
                    args.loss_weight_spectral, args.loss_weight_mel,
                    weight_correlation=0.0
                )
                
                pair_losses.append(pair_loss)
                pair_rms_input.append(rms_in)
                pair_rms_target.append(rms_tgt)
                pair_spectral.append(spec)
                pair_mel.append(mel)
            
            # Average losses across all 3 pairs
            loss = torch.stack(pair_losses).mean()
            rms_input_val = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in pair_rms_input]).mean()
            rms_target_val = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in pair_rms_target]).mean()
            spec_val = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in pair_spectral]).mean()
            mel_val = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in pair_mel]).mean()
            
            mean_novelty_loss = torch.stack(novelty_losses).mean()
            
            loss = loss + args.mask_reg_weight * mean_novelty_loss
            
            # Store first sample (decode first output for visualization)
            if batch_idx == 0:
                first_output_decoded = encodec_model.decoder(outputs_list[0][0:1].detach())  # First sample, first output
                first_input_audio = audio_inputs[0:1, :, 160000:416000].cpu()
                first_target_audio = audio_targets[0:1, :, 160000:416000].cpu()
                first_output_audio = first_output_decoded.cpu()
            
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
