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

# Import model and utilities
from adaptive_window_agent import AdaptiveWindowCreativeAgent
from dataset_wav_pairs import WavPairsDataset
from training.losses import combined_loss


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
    from audiocraft.models import CompressionModel
    
    encodec_model = CompressionModel.get_pretrained("facebook/encodec_24khz", device=device)
    encodec_model.set_target_bandwidth(bandwidth)
    encodec_model.eval()
    
    for param in encodec_model.parameters():
        param.requires_grad = False
    
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Bandwidth: {bandwidth}")
    
    return encodec_model


def compute_metrics(model, encodec_model, dataloader, device, args):
    """
    Run validation and compute all metrics
    
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
    
    with torch.no_grad():
        for batch_idx, (audio_inputs, audio_targets) in enumerate(tqdm(dataloader, desc="Validation")):
            audio_inputs = audio_inputs.to(device)
            audio_targets = audio_targets.to(device)
            
            # Encode
            encoded_inputs = encodec_model.encoder(audio_inputs)
            encoded_targets = encodec_model.encoder(audio_targets)
            
            # Forward pass
            outputs_list, novelty_losses, metadata = model(encoded_inputs, encoded_targets)
            
            # Compute per-pair losses
            pair_losses = []
            pair_rms_input = []
            pair_rms_target = []
            pair_spectral = []
            pair_mel = []
            pair_corr_penalty = []
            
            for idx, encoded_output in enumerate(outputs_list):
                # Decode output
                output_audio = encodec_model.decoder(encoded_output)
                
                # Extract center segments (256k samples)
                input_10sec = audio_inputs[:, :, 160000:416000]
                target_10sec = audio_targets[:, :, 160000:416000]
                
                # Compute loss
                pair_loss, rms_in, rms_tgt, spec, mel, corr = combined_loss(
                    output_audio, input_10sec, target_10sec,
                    getattr(args, 'loss_weight_input', 0.3),
                    getattr(args, 'loss_weight_target', 0.3),
                    getattr(args, 'loss_weight_spectral', 0.05),
                    getattr(args, 'loss_weight_mel', 0.05),
                    weight_correlation=getattr(args, 'corr_weight', 0.5)
                )
                
                pair_losses.append(pair_loss)
                pair_rms_input.append(rms_in)
                pair_rms_target.append(rms_tgt)
                pair_spectral.append(spec)
                pair_mel.append(mel)
                pair_corr_penalty.append(corr)
            
            # Average across pairs
            loss = torch.stack(pair_losses).mean()
            rms_input_val = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in pair_rms_input]).mean()
            rms_target_val = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in pair_rms_target]).mean()
            spec_val = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in pair_spectral]).mean()
            mel_val = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in pair_mel]).mean()
            corr_penalty_val = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in pair_corr_penalty]).mean()
            
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
                
                for b in range(audio_inputs.shape[0]):
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
            input_10sec = audio_inputs[:, :, 160000:416000]
            target_10sec = audio_targets[:, :, 160000:416000]
            
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


def generate_samples(model, encodec_model, dataloader, device, output_dir, num_samples=5):
    """Generate and save audio samples"""
    print(f"\n" + "="*80)
    print(f"GENERATING {num_samples} SAMPLES")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for sample_idx, (audio_inputs, audio_targets) in enumerate(dataloader):
            if sample_idx >= num_samples:
                break
            
            audio_inputs = audio_inputs.to(device)
            audio_targets = audio_targets.to(device)
            
            # Take first sample from batch
            audio_input = audio_inputs[0:1]
            audio_target = audio_targets[0:1]
            
            # Encode
            encoded_input = encodec_model.encoder(audio_input)
            encoded_target = encodec_model.encoder(audio_target)
            
            # Forward pass
            outputs_list, novelty_losses, metadata = model(encoded_input, encoded_target)
            
            # Decode all 3 pairs
            for pair_idx, encoded_output in enumerate(outputs_list):
                output_audio = encodec_model.decoder(encoded_output)
                
                # Convert to numpy
                input_np = audio_input.squeeze().cpu().numpy()
                target_np = audio_target.squeeze().cpu().numpy()
                output_np = output_audio.squeeze().cpu().numpy()
                
                # Save
                sf.write(output_dir / f"sample_{sample_idx:03d}_pair_{pair_idx}_input.wav", input_np, 24000)
                sf.write(output_dir / f"sample_{sample_idx:03d}_pair_{pair_idx}_target.wav", target_np, 24000)
                sf.write(output_dir / f"sample_{sample_idx:03d}_pair_{pair_idx}_output.wav", output_np, 24000)
            
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
        pairs_dir=Path(args.dataset),
        duration=24.0,
        sample_rate=24000
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Compute metrics
    metrics = compute_metrics(model, encodec_model, val_loader, args.device, train_args)
    
    # Print results
    print_results(metrics, epoch)
    
    # Generate samples
    generate_samples(model, encodec_model, val_loader, args.device, args.output_dir, args.num_samples)
    
    print("\n✅ Validation complete!")


if __name__ == "__main__":
    main()
