#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Pure GAN mode model with noisy validation samples

This script:
1. Takes N different clean input samples (variants)
2. For each variant, generates M predictions with noisy inputs/targets
3. Within a variant: --same_target controls if all predictions use same target
4. Saves in structure: variant_1/1_noisy_input.wav, 1_predicted.wav, 2_noisy_input.wav, etc.

Folder structure:
  variant_1/ (clean input #1 with noise)
    1_noisy_input.wav, 1_noisy_target.wav, 1_predicted.wav
    2_noisy_input.wav, 2_noisy_target.wav, 2_predicted.wav (different target if --same_target not used)
    ...
  variant_2/ (clean input #2 with noise)
    1_noisy_input.wav, 1_noisy_target.wav, 1_predicted.wav
    ...

Noise injection matches training:
  noisy_input = (1 - noise_fraction) × input + noise_fraction × (noise × input_std)
"""

import os
import torch
import soundfile as sf
import numpy as np
import encodec
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import signal

from model_simple_transformer import SimpleTransformer


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load trained Pure GAN model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    args = checkpoint.get('args', {})
    
    model = SimpleTransformer(
        encoding_dim=args.get('encoding_dim', 128),
        nhead=args.get('nhead', 8),
        num_layers=args.get('num_layers', 4),
        num_transformer_layers=args.get('num_transformer_layers', 1),
        dropout=args.get('dropout', 0.1),
        anti_cheating=args.get('anti_cheating', 0.0),
        use_creative_agent=args.get('use_creative_agent', False),
        use_compositional_agent=args.get('use_compositional_agent', False)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"  Cascade stages: {args.get('num_transformer_layers', 1)}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, args


def load_encodec(bandwidth=6.0, sample_rate=24000, device='cuda'):
    """Load EnCodec model"""
    print("\nLoading EnCodec model...")
    model = encodec.EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)
    model = model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"  Sample rate: {sample_rate} Hz, Bandwidth: {bandwidth}")
    return model


def generate_noisy_encoded(input_path, target_path, encodec_model, noise_fraction, device='cuda'):
    """
    Generate ONE noisy encoded pair from clean input/target using Pure GAN noise injection
    
    Returns:
        (noisy_encoded_input, noisy_encoded_target, input_std, target_std)
    """
    # Load audio files
    input_audio, sr = sf.read(input_path)
    target_audio, _ = sf.read(target_path)
    
    # Convert stereo to mono if needed
    if input_audio.ndim == 2:
        input_audio = input_audio.mean(axis=1)
    if target_audio.ndim == 2:
        target_audio = target_audio.mean(axis=1)
    
    # Convert to tensors
    input_audio_tensor = torch.from_numpy(input_audio).float().unsqueeze(0).unsqueeze(0).to(device)
    target_audio_tensor = torch.from_numpy(target_audio).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Encode clean versions
    with torch.no_grad():
        clean_encoded_input = encodec_model.encoder(input_audio_tensor)  # [1, 128, T]
        clean_encoded_target = encodec_model.encoder(target_audio_tensor)  # [1, 128, T]
        
        # Compute per-sample std (same as training)
        input_std = clean_encoded_input.std(dim=(1, 2), keepdim=True)  # [1, 1, 1]
        target_std = clean_encoded_target.std(dim=(1, 2), keepdim=True)  # [1, 1, 1]
        
        # Generate random noise scaled by per-sample std
        input_noise = torch.randn_like(clean_encoded_input) * input_std
        target_noise = torch.randn_like(clean_encoded_target) * target_std
        
        # Interpolate: (1-α) × clean + α × noise
        alpha = noise_fraction
        noisy_encoded_input = (1.0 - alpha) * clean_encoded_input + alpha * input_noise
        noisy_encoded_target = (1.0 - alpha) * clean_encoded_target + alpha * target_noise
    
    return noisy_encoded_input, noisy_encoded_target, input_std.item(), target_std.item()


def generate_audio_from_noisy(model, encodec_model, noisy_encoded_input, noisy_encoded_target,
                              num_cascade_stages, device='cuda'):
    """Generate audio from noisy encoded inputs using cascade model"""
    with torch.no_grad():
        # Transform with cascade model
        if num_cascade_stages > 1:
            result = model(noisy_encoded_input, noisy_encoded_target)
            encoded_output = result[0] if isinstance(result, tuple) else result
        else:
            result = model(noisy_encoded_input)
            encoded_output = result[0] if isinstance(result, tuple) else result
        
        # Decode to audio
        predicted_audio = encodec_model.decoder(encoded_output)
        predicted_audio = predicted_audio.squeeze().cpu().numpy().astype(np.float32)
    
    return predicted_audio


def save_audio(audio, path, sample_rate=24000):
    """Save audio to file"""
    audio = np.array(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, audio, sample_rate, subtype='PCM_16')


def visualize_waveforms(input_audio, target_audio, predicted_audio, output_path, 
                       noise_fraction, sample_rate=24000):
    """Visualize waveforms"""
    duration = len(input_audio) / sample_rate
    time = np.linspace(0, duration, len(input_audio))
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    axes[0].plot(time, input_audio, color='blue', linewidth=0.5, alpha=0.7)
    axes[0].set_title(f'Noisy Input ({noise_fraction*100:.0f}% noise)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_ylim(-1.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time, target_audio, color='green', linewidth=0.5, alpha=0.7)
    axes[1].set_title(f'Noisy Target ({noise_fraction*100:.0f}% noise)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(time, predicted_audio, color='red', linewidth=0.5, alpha=0.7)
    axes[2].set_title('Predicted Audio (Pure GAN Model)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_ylim(-1.0, 1.0)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_spectrograms(input_audio, target_audio, predicted_audio, output_path, 
                          noise_fraction, sample_rate=24000):
    """Visualize spectrograms"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    n_fft = 2048
    hop_length = 512
    
    # Input spectrogram
    D_input = np.abs(plt.specgram(input_audio, Fs=sample_rate, NFFT=n_fft, 
                                    noverlap=n_fft-hop_length, cmap='viridis')[0])
    axes[0].clear()
    im0 = axes[0].imshow(10 * np.log10(D_input + 1e-10), aspect='auto', origin='lower', 
                         cmap='viridis', extent=[0, len(input_audio)/sample_rate, 0, sample_rate/2])
    axes[0].set_title(f'Noisy Input Spectrogram ({noise_fraction*100:.0f}% noise)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_ylim(0, 8000)
    plt.colorbar(im0, ax=axes[0], label='Power (dB)')
    
    # Target spectrogram
    D_target = np.abs(plt.specgram(target_audio, Fs=sample_rate, NFFT=n_fft, 
                                     noverlap=n_fft-hop_length, cmap='viridis')[0])
    axes[1].clear()
    im1 = axes[1].imshow(10 * np.log10(D_target + 1e-10), aspect='auto', origin='lower', 
                         cmap='viridis', extent=[0, len(target_audio)/sample_rate, 0, sample_rate/2])
    axes[1].set_title(f'Noisy Target Spectrogram ({noise_fraction*100:.0f}% noise)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_ylim(0, 8000)
    plt.colorbar(im1, ax=axes[1], label='Power (dB)')
    
    # Predicted spectrogram
    D_pred = np.abs(plt.specgram(predicted_audio, Fs=sample_rate, NFFT=n_fft, 
                                   noverlap=n_fft-hop_length, cmap='viridis')[0])
    axes[2].clear()
    im2 = axes[2].imshow(10 * np.log10(D_pred + 1e-10), aspect='auto', origin='lower', 
                         cmap='viridis', extent=[0, len(predicted_audio)/sample_rate, 0, sample_rate/2])
    axes[2].set_title('Predicted Audio Spectrogram (Pure GAN)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_ylim(0, 8000)
    plt.colorbar(im2, ax=axes[2], label='Power (dB)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test Pure GAN model with noisy validation samples')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_GAN/best_model.pt',
                       help='Path to Pure GAN checkpoint')
    parser.add_argument('--val_folder', type=str, default='dataset_pairs_wav/val',
                       help='Validation dataset folder')
    parser.add_argument('--noise_fraction', type=float, default=0.23,
                       help='Fraction of noise to add (0.0-1.0, default: 0.23 for 23%% noise)')
    parser.add_argument('--num_variants', type=int, default=5,
                       help='Number of different clean input samples to process')
    parser.add_argument('--num_predictions', type=int, default=1,
                       help='Number of predictions per variant (default: 1)')
    parser.add_argument('--same_target', action='store_true',
                       help='Use same target for all predictions within a variant')
    parser.add_argument('--shuffle', action='store_true',
                       help='Use random targets instead of matched continuations')
    parser.add_argument('--cache_folder', type=str, default='cache_noisy',
                       help='Cache folder for noisy audio files')
    parser.add_argument('--output_folder', type=str, default='inference_gan_outputs',
                       help='Output folder for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Validate noise_fraction
    if not 0.0 <= args.noise_fraction <= 1.0:
        print(f"Error: noise_fraction must be between 0.0 and 1.0 (got {args.noise_fraction})")
        return
    
    # Create folders
    os.makedirs(args.cache_folder, exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)
    
    print("="*80)
    print("PURE GAN MODE VALIDATION WITH NOISY SAMPLES")
    print("="*80)
    
    # Load models
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model, model_args = load_checkpoint(args.checkpoint, device)
    encodec_model = load_encodec(device=device)
    
    # Get cascade configuration
    num_cascade_stages = model_args.get('num_transformer_layers', 1)
    
    # Find validation samples
    val_files = sorted([f for f in os.listdir(args.val_folder) if f.endswith('_input.wav')])
    all_output_files = sorted([f for f in os.listdir(args.val_folder) if f.endswith('_output.wav')])
    
    if len(val_files) == 0:
        print(f"\nError: No validation samples found in {args.val_folder}")
        return
    
    # Filter to complete pairs
    valid_pairs = []
    for input_file in val_files:
        output_file = input_file.replace('_input.wav', '_output.wav')
        input_path = os.path.join(args.val_folder, input_file)
        output_path = os.path.join(args.val_folder, output_file)
        
        if os.path.exists(input_path) and os.path.exists(output_path):
            valid_pairs.append(input_file)
    
    if len(valid_pairs) == 0:
        print(f"\nError: No complete pairs found in {args.val_folder}")
        return
    
    num_variants = min(args.num_variants, len(valid_pairs))
    
    print(f"\nConfiguration:")
    print(f"  Variants (clean inputs): {num_variants} (from {len(valid_pairs)} available)")
    print(f"  Predictions per variant: {args.num_predictions}")
    print(f"  Noise fraction: {args.noise_fraction:.2%} ({(1-args.noise_fraction)*100:.0f}% music + {args.noise_fraction*100:.0f}% noise)")
    print(f"  Total inference runs: {num_variants * args.num_predictions}")
    print(f"  Cascade stages: {num_cascade_stages}")
    print(f"  Same target per variant: {'YES' if args.same_target else 'NO (random each time)'}")
    print(f"  Shuffle targets: {'ENABLED (random pairs)' if args.shuffle else 'DISABLED (matched pairs)'}")
    
    # Process each variant (clean input sample)
    for variant_idx in range(num_variants):
        print("\n" + "="*80)
        print(f"VARIANT {variant_idx + 1}/{num_variants} (Clean Input Sample #{variant_idx + 1})")
        print("="*80)
        
        input_file = valid_pairs[variant_idx]
        input_path = os.path.join(args.val_folder, input_file)
        sample_name = input_file.replace('_input.wav', '')
        
        print(f"Clean input: {input_file}")
        
        # Create output folder for this variant
        variant_folder = os.path.join(args.output_folder, f"variant_{variant_idx+1}")
        cache_variant_folder = os.path.join(args.cache_folder, f"variant_{variant_idx+1}")
        os.makedirs(variant_folder, exist_ok=True)
        os.makedirs(cache_variant_folder, exist_ok=True)
        
        # Select target once if --same_target is enabled
        if args.same_target:
            if args.shuffle:
                selected_target = np.random.choice(all_output_files)
                print(f"Selected target (same for all predictions): {selected_target} 🎲")
            else:
                selected_target = input_file.replace('_input.wav', '_output.wav')
                print(f"Selected target (same for all predictions): {selected_target}")
        
        # Generate multiple predictions for this variant
        print(f"\n🎵 Generating {args.num_predictions} predictions for this variant...")
        
        for pred_idx in range(args.num_predictions):
            print(f"\n  Prediction {pred_idx + 1}/{args.num_predictions}:")
            
            # Select target for this prediction
            if args.same_target:
                output_file = selected_target
            else:
                if args.shuffle:
                    output_file = np.random.choice(all_output_files)
                    print(f"    🎲 Random target: {output_file}")
                else:
                    output_file = input_file.replace('_input.wav', '_output.wav')
                    print(f"    Matched target: {output_file}")
            
            target_path = os.path.join(args.val_folder, output_file)
            
            # Generate noisy encoded pair
            noisy_input, noisy_target, input_std, target_std = generate_noisy_encoded(
                input_path, target_path, encodec_model, args.noise_fraction, device
            )
            
            if pred_idx == 0:
                print(f"    Noise statistics:")
                print(f"      Input:  {(1-args.noise_fraction)*100:.1f}% music + {args.noise_fraction*100:.1f}% noise")
                print(f"      Target: {(1-args.noise_fraction)*100:.1f}% music + {args.noise_fraction*100:.1f}% noise")
                print(f"      Input std: {input_std:.4f}, Target std: {target_std:.4f}")
            
            # Generate audio
            predicted_audio = generate_audio_from_noisy(
                model, encodec_model, noisy_input, noisy_target,
                num_cascade_stages, device
            )
            
            # Decode noisy inputs for visualization
            with torch.no_grad():
                noisy_input_audio = encodec_model.decoder(noisy_input).squeeze().cpu().numpy()
                noisy_target_audio = encodec_model.decoder(noisy_target).squeeze().cpu().numpy()
            
            # Save to cache
            save_audio(noisy_input_audio, os.path.join(cache_variant_folder, f'{pred_idx+1}_noisy_input.wav'))
            save_audio(noisy_target_audio, os.path.join(cache_variant_folder, f'{pred_idx+1}_noisy_target.wav'))
            
            # Save audio files with prediction index prefix
            save_audio(noisy_input_audio, os.path.join(variant_folder, f'{pred_idx+1}_noisy_input.wav'))
            save_audio(noisy_target_audio, os.path.join(variant_folder, f'{pred_idx+1}_noisy_target.wav'))
            save_audio(predicted_audio, os.path.join(variant_folder, f'{pred_idx+1}_predicted.wav'))
            
            # Generate visualizations
            visualize_waveforms(
                noisy_input_audio, noisy_target_audio, predicted_audio,
                os.path.join(variant_folder, f'{pred_idx+1}_waveforms.png'),
                args.noise_fraction
            )
            
            visualize_spectrograms(
                noisy_input_audio, noisy_target_audio, predicted_audio,
                os.path.join(variant_folder, f'{pred_idx+1}_spectrograms.png'),
                args.noise_fraction
            )
            
            # Compute RMS
            output_rms = np.sqrt(np.mean(predicted_audio ** 2))
            input_rms = np.sqrt(np.mean(noisy_input_audio ** 2))
            print(f"    Output RMS: {output_rms:.6f}, Input RMS: {input_rms:.6f}")
        
        print(f"\n  ✓ Variant {variant_idx+1} complete: {args.num_predictions} predictions saved to {variant_folder}/")
    
    print("\n" + "="*80)
    print("PURE GAN VALIDATION COMPLETE")
    print("="*80)
    print(f"\nProcessed {num_variants} variants × {args.num_predictions} predictions = {num_variants * args.num_predictions} total outputs")
    print(f"\nFolder structure:")
    print(f"  {args.output_folder}/")
    print(f"    variant_1/          - Clean input sample #1 with noise")
    print(f"      1_noisy_input.wav     - Noisy input ({args.noise_fraction*100:.0f}% noise)")
    print(f"      1_noisy_target.wav    - Noisy target ({args.noise_fraction*100:.0f}% noise)")
    print(f"      1_predicted.wav       - Model prediction")
    print(f"      1_waveforms.png       - Visualization")
    print(f"      1_spectrograms.png    - Visualization")
    if args.num_predictions > 1:
        print(f"      2_noisy_input.wav     - Second prediction (different target)")
        print(f"      2_noisy_target.wav")
        print(f"      2_predicted.wav")
        print(f"      ...")
    print(f"    variant_2/          - Clean input sample #2 with noise")
    print(f"      (same structure)")
    print(f"\nCached noisy audio in: {args.cache_folder}/variant_N/")
    print("="*80)


if __name__ == "__main__":
    main()
