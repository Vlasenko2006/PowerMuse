#!/usr/bin/env python3
"""
Validation script for hybrid adaptive window training

Loads a trained checkpoint and validates on test set:
- Computes all metrics (RMS, spectral, correlation, etc.)
- Generates audio samples
- Analyzes window selection statistics
- Evaluates ratio supervision effectiveness
"""

import os
import sys
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Import model and utilities
from adaptive_window_agent import AdaptiveWindowCreativeAgent
from dataset_wav_pairs import WavPairsDataset
from encodec import EncodecModel


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load trained hybrid model from checkpoint"""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get configuration
    args = checkpoint.get('args', argparse.Namespace())
    epoch = checkpoint.get('epoch', 0)
    
    print(f"  Epoch: {epoch}")
    print(f"  Train loss: {checkpoint.get('loss', 'N/A')}")
    
    # Create model
    model = AdaptiveWindowCreativeAgent(
        encoding_dim=getattr(args, 'encoding_dim', 128),
        num_pairs=getattr(args, 'num_pairs', 3)
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, args, epoch


def load_encodec(bandwidth=6.0, sample_rate=24000, device='cuda'):
    """Load EnCodec model"""
    print("\nLoading EnCodec...")
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)
    model = model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Bandwidth: {bandwidth}")
    
    return model


def encode_audio_batch(audio_batch, encodec_model, target_frames=1200):
    """
    Encode batch of audio with EnCodec and resample to target frames.
    
    Args:
        audio_batch: [B, 1, samples] - mono audio (24 sec = 576,000 samples)
        encodec_model: EnCodec model
        target_frames: Target number of frames (1200 = 16 sec equivalent)
    
    Returns:
        encoded: [B, 128, target_frames]
    """
    with torch.no_grad():
        latents = encodec_model.encoder(audio_batch)  # [B, 128, T]
        
        # Resample to target frames if needed
        if latents.shape[-1] != target_frames:
            latents = F.interpolate(
                latents,
                size=target_frames,
                mode='linear',
                align_corners=False
            )
    
    return latents


def compute_metrics(model, encodec_model, dataloader, device, args, max_batches=None):
    """
    Run validation and compute all metrics
    
    Args:
        max_batches: If provided, process only this many batches
    
    Returns:
        Dictionary with averaged metrics
    """
    print("\n" + "="*80)
    print("RUNNING VALIDATION")
    print("="*80)
    
    metrics = {
        'loss': 0.0,
        'rms_input': 0.0,
        'rms_target': 0.0,
        'spectral': 0.0,
        'mel': 0.0,
        'novelty': 0.0,
        'corr_penalty': 0.0,
        'ratio_diversity': 0.0,
        'reconstruction': 0.0,
        'output_input_corr': 0.0,
        'output_target_corr': 0.0,
        # Window stats
        'pair0_ratio_in': 0.0,
        'pair0_ratio_tgt': 0.0,
        'pair1_ratio_in': 0.0,
        'pair1_ratio_tgt': 0.0,
        'pair2_ratio_in': 0.0,
        'pair2_ratio_tgt': 0.0,
        'pair0_tonality': 0.0,
        'pair1_tonality': 0.0,
        'pair2_tonality': 0.0,
    }
    
    num_batches = 0
    
    total_batches = max_batches if max_batches is not None else len(dataloader)
    
    with torch.no_grad():
        for batch_idx, (encoded_inputs_1800, audio_targets) in enumerate(tqdm(dataloader, desc="Validation", total=total_batches)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            # Dataset returns: (encoded_input at 1800 frames, raw_audio_target at 576k samples)
            audio_targets = audio_targets.to(device)
            
            # Resample encoded inputs from 1800 to 1200 frames (same as training)
            encoded_inputs_1800 = encoded_inputs_1800.to(device)
            encoded_inputs = F.interpolate(
                encoded_inputs_1800,
                size=1200,
                mode='linear',
                align_corners=False
            )
            
            # Encode targets to 1200 frames (same as training)
            encoded_targets = encode_audio_batch(audio_targets, encodec_model, target_frames=1200)
            
            # Forward pass
            outputs_list, novelty_losses, metadata = model(encoded_inputs, encoded_targets)
            
            # Compute per-pair RMS errors (same as training)
            pair_rms_input = []
            pair_rms_target = []
            
            for idx, encoded_output in enumerate(outputs_list):
                # Decode output: 800 frames → 256k samples
                output_audio = encodec_model.decoder(encoded_output)
                
                # Extract center 256k samples from raw audio (same as training)
                # Center offset: (576000 - 256000) / 2 = 160000
                target_10sec = audio_targets[:, :, 160000:416000]  # [B, 1, 256000]
                
                # For input: decode the 1200-frame encoded input, then extract center
                # But we don't have raw input audio! We only have encoded input.
                # So decode it first: 1200 frames → 384k samples
                input_audio_full = encodec_model.decoder(encoded_inputs)  # [B, 1, 384000]
                # Extract center 256k from 384k: offset = (384000 - 256000) / 2 = 64000
                input_10sec = input_audio_full[:, :, 64000:320000]  # [B, 1, 256000]
                
                # Compute simple RMS errors
                rms_in = F.mse_loss(output_audio, input_10sec).sqrt()
                rms_tgt = F.mse_loss(output_audio, target_10sec).sqrt()
                
                pair_rms_input.append(rms_in)
                pair_rms_target.append(rms_tgt)
            
            # Average across pairs
            rms_input_val = torch.stack(pair_rms_input).mean()
            rms_target_val = torch.stack(pair_rms_target).mean()
            loss = (rms_input_val + rms_target_val) / 2
            spec_val = torch.tensor(0.0)  # Not computed in validation
            mel_val = torch.tensor(0.0)  # Not computed in validation
            corr_penalty_val = torch.tensor(0.0)  # Not computed in validation
            
            # Novelty loss
            mean_novelty_loss = torch.stack(novelty_losses).mean()
            
            # Ratio diversity loss
            if 'window_params' in metadata:
                all_ratios_input = [p['ratio_input'] for p in metadata['window_params']]
                all_ratios_target = [p['ratio_target'] for p in metadata['window_params']]
                
                ratios_input_stacked = torch.stack(all_ratios_input, dim=0)
                ratios_target_stacked = torch.stack(all_ratios_target, dim=0)
                
                ratio_variance_input = torch.var(ratios_input_stacked, dim=0).mean()
                ratio_variance_target = torch.var(ratios_target_stacked, dim=0).mean()
                
                ratio_diversity_loss = -(ratio_variance_input + ratio_variance_target)
            else:
                ratio_diversity_loss = torch.tensor(0.0, device=device)
            
            # Reconstruction loss
            reconstruction_losses = []
            for idx, (encoded_output, params) in enumerate(zip(outputs_list, metadata['window_params'])):
                output_audio_decoded = encodec_model.decoder(encoded_output)
                
                original_input_windows = []
                original_target_windows = []
                
                for b in range(encoded_inputs.shape[0]):  # Use encoded_inputs shape
                    start_in = int(params['start_input'][b].item())
                    start_tgt = int(params['start_target'][b].item())
                    
                    start_in = max(0, min(start_in, encoded_inputs.shape[2] - 800))
                    start_tgt = max(0, min(start_tgt, encoded_targets.shape[2] - 800))
                    
                    orig_in = encoded_inputs[b:b+1, :, start_in:start_in+800]
                    orig_tgt = encoded_targets[b:b+1, :, start_tgt:start_tgt+800]
                    
                    original_input_windows.append(orig_in)
                    original_target_windows.append(orig_tgt)
                
                orig_in_batch = torch.cat(original_input_windows, dim=0)
                orig_tgt_batch = torch.cat(original_target_windows, dim=0)
                
                orig_in_audio = encodec_model.decoder(orig_in_batch)
                orig_tgt_audio = encodec_model.decoder(orig_tgt_batch)
                
                recon_loss_input = F.mse_loss(output_audio_decoded, orig_in_audio)
                recon_loss_target = F.mse_loss(output_audio_decoded, orig_tgt_audio)
                
                recon_loss = (recon_loss_input + recon_loss_target) / 2
                reconstruction_losses.append(recon_loss)
            
            mean_reconstruction_loss = torch.stack(reconstruction_losses).mean()
            
            # Correlation analysis
            output_audio_first = encodec_model.decoder(outputs_list[0])
            # Decode inputs and extract center
            input_audio_full = encodec_model.decoder(encoded_inputs)
            input_10sec = input_audio_full[:, :, 64000:320000]  # Center 256k from 384k
            target_10sec = audio_targets[:, :, 160000:416000]  # Center 256k from 576k
            
            output_flat = output_audio_first.reshape(-1)
            input_flat = input_10sec.reshape(-1)
            target_flat = target_10sec.reshape(-1)
            
            output_input_corr = torch.corrcoef(torch.stack([output_flat, input_flat]))[0, 1]
            output_target_corr = torch.corrcoef(torch.stack([output_flat, target_flat]))[0, 1]
            
            # Accumulate metrics
            metrics['loss'] += loss.item()
            metrics['rms_input'] += rms_input_val.item()
            metrics['rms_target'] += rms_target_val.item()
            metrics['spectral'] += spec_val.item()
            metrics['mel'] += mel_val.item()
            metrics['novelty'] += mean_novelty_loss.item()
            metrics['corr_penalty'] += corr_penalty_val.item()
            metrics['ratio_diversity'] += ratio_diversity_loss.item()
            metrics['reconstruction'] += mean_reconstruction_loss.item()
            metrics['output_input_corr'] += output_input_corr.item()
            metrics['output_target_corr'] += output_target_corr.item()
            
            # Window stats
            if 'pairs' in metadata:
                metrics['pair0_ratio_in'] += metadata['pairs'][0].get('ratio_input_mean', 0.0)
                metrics['pair0_ratio_tgt'] += metadata['pairs'][0].get('ratio_target_mean', 0.0)
                metrics['pair1_ratio_in'] += metadata['pairs'][1].get('ratio_input_mean', 0.0)
                metrics['pair1_ratio_tgt'] += metadata['pairs'][1].get('ratio_target_mean', 0.0)
                metrics['pair2_ratio_in'] += metadata['pairs'][2].get('ratio_input_mean', 0.0)
                metrics['pair2_ratio_tgt'] += metadata['pairs'][2].get('ratio_target_mean', 0.0)
                metrics['pair0_tonality'] += metadata['pairs'][0].get('tonality_strength_mean', 0.0)
                metrics['pair1_tonality'] += metadata['pairs'][1].get('tonality_strength_mean', 0.0)
                metrics['pair2_tonality'] += metadata['pairs'][2].get('tonality_strength_mean', 0.0)
            
            num_batches += 1
    
    # Average metrics
    for key in metrics:
        metrics[key] /= num_batches
    
    return metrics


def visualize_waveforms(input_audio, target_audio, output_audio, output_path, sample_rate=24000):
    """Visualize input, target, and output waveforms"""
    duration_input = len(input_audio) / sample_rate
    duration_target = len(target_audio) / sample_rate
    duration_output = len(output_audio) / sample_rate
    
    time_input = np.linspace(0, duration_input, len(input_audio))
    time_target = np.linspace(0, duration_target, len(target_audio))
    time_output = np.linspace(0, duration_output, len(output_audio))
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot input
    axes[0].plot(time_input, input_audio, color='blue', linewidth=0.5, alpha=0.7)
    axes[0].set_title('Input Audio', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].set_xlim(0, duration_input)
    axes[0].set_ylim(-1.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Plot target
    axes[1].plot(time_target, target_audio, color='green', linewidth=0.5, alpha=0.7)
    axes[1].set_title('Target Audio (Ground Truth)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Amplitude', fontsize=12)
    axes[1].set_xlim(0, duration_target)
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Plot output
    axes[2].plot(time_output, output_audio, color='red', linewidth=0.5, alpha=0.7)
    axes[2].set_title('Model Output', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (seconds)', fontsize=12)
    axes[2].set_ylabel('Amplitude', fontsize=12)
    axes[2].set_xlim(0, duration_output)
    axes[2].set_ylim(-1.0, 1.0)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_spectrograms(input_audio, target_audio, output_audio, output_path, sample_rate=24000):
    """Visualize spectrograms of input, target, and output audio"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    n_fft = 2048
    hop_length = 512
    
    # Input spectrogram
    D_input = np.abs(plt.specgram(input_audio, Fs=sample_rate, NFFT=n_fft, 
                                    noverlap=n_fft-hop_length, cmap='viridis')[0])
    axes[0].clear()
    im0 = axes[0].imshow(10 * np.log10(D_input + 1e-10), aspect='auto', origin='lower', 
                         cmap='viridis', extent=[0, len(input_audio)/sample_rate, 0, sample_rate/2])
    axes[0].set_title('Input Audio Spectrogram', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[0].set_ylim(0, 8000)
    plt.colorbar(im0, ax=axes[0], label='Power (dB)')
    
    # Target spectrogram
    D_target = np.abs(plt.specgram(target_audio, Fs=sample_rate, NFFT=n_fft, 
                                     noverlap=n_fft-hop_length, cmap='viridis')[0])
    axes[1].clear()
    im1 = axes[1].imshow(10 * np.log10(D_target + 1e-10), aspect='auto', origin='lower', 
                         cmap='viridis', extent=[0, len(target_audio)/sample_rate, 0, sample_rate/2])
    axes[1].set_title('Target Audio Spectrogram (Ground Truth)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[1].set_ylim(0, 8000)
    plt.colorbar(im1, ax=axes[1], label='Power (dB)')
    
    # Output spectrogram
    D_output = np.abs(plt.specgram(output_audio, Fs=sample_rate, NFFT=n_fft, 
                                   noverlap=n_fft-hop_length, cmap='viridis')[0])
    axes[2].clear()
    im2 = axes[2].imshow(10 * np.log10(D_output + 1e-10), aspect='auto', origin='lower', 
                         cmap='viridis', extent=[0, len(output_audio)/sample_rate, 0, sample_rate/2])
    axes[2].set_title('Model Output Spectrogram', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (seconds)', fontsize=12)
    axes[2].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[2].set_ylim(0, 8000)
    plt.colorbar(im2, ax=axes[2], label='Power (dB)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_samples(model, encodec_model, dataloader, device, output_dir, num_samples=5):
    """Generate and save audio samples"""
    print(f"\n" + "="*80)
    print(f"GENERATING {num_samples} SAMPLES")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for sample_idx, (encoded_inputs_1800, audio_targets) in enumerate(dataloader):
            if sample_idx >= num_samples:
                break
            
            # Dataset returns: (encoded_input at 1800 frames, raw_audio_target at 576k samples)
            audio_targets = audio_targets.to(device)
            audio_target = audio_targets[0:1]
            
            # Take first sample and resample from 1800 to 1200 frames
            encoded_inputs_1800 = encoded_inputs_1800.to(device)
            encoded_input = encoded_inputs_1800[0:1]
            encoded_input = F.interpolate(
                encoded_input,
                size=1200,
                mode='linear',
                align_corners=False
            )
            
            # Encode target to 1200 frames
            encoded_target = encode_audio_batch(audio_target, encodec_model, target_frames=1200)
            
            # Forward pass
            outputs_list, novelty_losses, metadata = model(encoded_input, encoded_target)
            
            # Decode input for saving
            audio_input = encodec_model.decoder(encoded_input)
            
            # Decode all 3 pairs and generate visualizations
            for pair_idx, encoded_output in enumerate(outputs_list):
                output_audio = encodec_model.decoder(encoded_output)
                
                # Convert to numpy
                input_np = audio_input.squeeze().cpu().numpy()
                target_np = audio_target.squeeze().cpu().numpy()
                output_np = output_audio.squeeze().cpu().numpy()
                
                # Save audio files
                sf.write(output_dir / f"sample_{sample_idx:03d}_pair_{pair_idx}_input.wav", input_np, 24000)
                sf.write(output_dir / f"sample_{sample_idx:03d}_pair_{pair_idx}_target.wav", target_np, 24000)
                sf.write(output_dir / f"sample_{sample_idx:03d}_pair_{pair_idx}_output.wav", output_np, 24000)
                
                # Generate waveform visualization
                visualize_waveforms(
                    input_np, target_np, output_np,
                    output_dir / f"sample_{sample_idx:03d}_pair_{pair_idx}_waveforms.png"
                )
                
                # Generate spectrogram visualization
                visualize_spectrograms(
                    input_np, target_np, output_np,
                    output_dir / f"sample_{sample_idx:03d}_pair_{pair_idx}_spectrograms.png"
                )
            
            # Print window info
            print(f"\nSample {sample_idx}:")
            for i, pair in enumerate(metadata['pairs']):
                print(f"  Pair {i}: Input ratio={pair.get('ratio_input_mean', 0):.2f}x, "
                      f"Target ratio={pair.get('ratio_target_mean', 0):.2f}x, "
                      f"Tonality={pair.get('tonality_strength_mean', 0):.2f}")
    
    print(f"\n✓ Saved samples to: {output_dir}")


def print_results(metrics, epoch):
    """Print validation results"""
    print("\n" + "="*80)
    print(f"VALIDATION RESULTS (Epoch {epoch})")
    print("="*80)
    
    print(f"\n📊 Loss Metrics:")
    print(f"  Total Loss:       {metrics['loss']:.6f}")
    print(f"  RMS Input:        {metrics['rms_input']:.6f}")
    print(f"  RMS Target:       {metrics['rms_target']:.6f}")
    print(f"  Spectral:         {metrics['spectral']:.6f}")
    print(f"  Mel:              {metrics['mel']:.6f}")
    print(f"  Novelty:          {metrics['novelty']:.6f}")
    print(f"  Corr Penalty:     {metrics['corr_penalty']:.6f}")
    
    print(f"\n🎯 Ratio Supervision:")
    print(f"  Ratio Diversity:  {metrics['ratio_diversity']:.6f}")
    print(f"  Reconstruction:   {metrics['reconstruction']:.6f}")
    
    print(f"\n📈 Correlations:")
    print(f"  Output→Input:     {metrics['output_input_corr']:.4f}")
    print(f"  Output→Target:    {metrics['output_target_corr']:.4f}")
    
    print(f"\n🔧 Window Selection (Averaged):")
    print(f"  Pair 0: Input={metrics['pair0_ratio_in']:.2f}x, Target={metrics['pair0_ratio_tgt']:.2f}x, Tonality={metrics['pair0_tonality']:.2f}")
    print(f"  Pair 1: Input={metrics['pair1_ratio_in']:.2f}x, Target={metrics['pair1_ratio_tgt']:.2f}x, Tonality={metrics['pair1_tonality']:.2f}")
    print(f"  Pair 2: Input={metrics['pair2_ratio_in']:.2f}x, Target={metrics['pair2_ratio_tgt']:.2f}x, Tonality={metrics['pair2_tonality']:.2f}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Validate hybrid adaptive window model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_24/hybrid_epoch_30.pt', help='Path to checkpoint')
    parser.add_argument('--dataset', type=str, default='dataset_pairs_wav_24sec/val', help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of audio samples to generate')
    parser.add_argument('--output_dir', type=str, default='validation_samples', help='Output directory for samples')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Load model
    model, train_args, epoch = load_checkpoint(args.checkpoint, args.device)
    
    # Load EnCodec
    encodec_model = load_encodec(device=args.device)
    
    # Load validation dataset
    print(f"\nLoading validation dataset from: {args.dataset}")
    val_dataset = WavPairsDataset(
        data_folder=args.dataset,
        encodec_model=encodec_model,
        device=args.device,
        shuffle_targets=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0  # Set to 0 for validation (dataset has encodec_model)
    )
    
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Calculate max batches needed for num_samples
    max_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    print(f"  Processing {max_batches} batches for {args.num_samples} samples\n")
    
    # Compute metrics
    metrics = compute_metrics(model, encodec_model, val_loader, args.device, train_args, max_batches=max_batches)
    
    # Print results
    print_results(metrics, epoch)
    
    # Generate samples
    generate_samples(model, encodec_model, val_loader, args.device, args.output_dir, args.num_samples)
    
    print("\n✅ Validation complete!")


if __name__ == "__main__":
    main()
