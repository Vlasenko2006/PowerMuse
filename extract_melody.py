#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract melody (vocal/lead) from audio using source separation.

Uses spleeter or basic spectral median filtering to isolate melodic components.
"""

import os
import argparse
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy.ndimage import median_filter


def extract_melody_simple(audio_path, output_path, percentile=50, sr=24000):
    """
    Extract melody using spectral median filtering.
    Isolates pitched/tonal content in the mid-high frequency range.
    
    Args:
        audio_path: Input audio file path
        output_path: Output file path
        percentile: Threshold percentile for melody extraction (30-70)
        sr: Target sample rate
    """
    print(f"Loading: {audio_path}")
    
    # Load audio
    y, orig_sr = librosa.load(audio_path, sr=sr, mono=True)
    print(f"  Duration: {len(y)/sr:.2f}s, SR: {sr} Hz")
    
    # Compute STFT
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    mag = np.abs(D)
    phase = np.angle(D)
    
    print(f"  Extracting melody (percentile={percentile})...")
    
    # Separate using harmonic component + median filtering
    # Get harmonic component (pitched content)
    H, P = librosa.decompose.hpss(D, margin=2.0)
    H_mag = np.abs(H)
    
    # Apply median filter to emphasize sustained notes (melody)
    # Median filter along time axis to keep stable frequencies
    H_mag_filtered = median_filter(H_mag, size=(1, 31))  # 31 frames ~= 0.7s
    
    # Keep only strong harmonic content above threshold
    threshold = np.percentile(H_mag_filtered, percentile)
    mask = (H_mag_filtered > threshold).astype(float)
    
    # Smooth mask to avoid artifacts
    mask = median_filter(mask, size=(3, 3))
    
    # Apply mask
    D_melody = H * mask
    
    # Convert back to time domain
    y_melody = librosa.istft(D_melody)
    
    # Normalize
    y_melody = y_melody / (np.abs(y_melody).max() + 1e-8)
    y_melody = np.clip(y_melody, -0.99, 0.99)
    
    # Compute energy ratio
    orig_energy = np.sqrt(np.mean(y ** 2))
    melody_energy = np.sqrt(np.mean(y_melody ** 2))
    ratio = melody_energy / (orig_energy + 1e-8)
    
    print(f"  Original RMS: {orig_energy:.4f}")
    print(f"  Melody RMS: {melody_energy:.4f}")
    print(f"  Energy ratio: {ratio:.2%}")
    
    # Save output
    save_audio(y_melody, output_path, sr)
    
    return y_melody


def save_audio(y, output_path, sr):
    """Save audio to file (MP3 or WAV)."""
    if output_path.endswith('.mp3'):
        try:
            from pydub import AudioSegment
            temp_wav = output_path.replace('.mp3', '_temp.wav')
            sf.write(temp_wav, y, sr)
            audio = AudioSegment.from_wav(temp_wav)
            audio.export(output_path, format='mp3', bitrate='192k')
            os.remove(temp_wav)
            print(f"  ✓ Saved: {output_path}")
        except ImportError:
            print("  ⚠️  pydub not installed, saving as WAV instead")
            output_path = output_path.replace('.mp3', '.wav')
            sf.write(output_path, y, sr)
            print(f"  ✓ Saved: {output_path}")
    else:
        sf.write(output_path, y, sr)
        print(f"  ✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract melody from audio files')
    parser.add_argument('--input', type=str, required=True,
                       help='Input audio file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: input_melody.mp3)')
    parser.add_argument('--percentile', type=float, default=50,
                       help='Threshold percentile for melody extraction (30-70)')
    parser.add_argument('--sr', type=int, default=24000,
                       help='Sample rate')
    parser.add_argument('--batch', action='store_true',
                       help='Process all files in input directory')
    parser.add_argument('--output_dir', type=str, default='melody_extracted',
                       help='Output directory for batch mode')
    
    args = parser.parse_args()
    
    if args.batch:
        input_dir = Path(args.input)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("="*80)
        print("BATCH MELODY EXTRACTION")
        print("="*80)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Percentile: {args.percentile}")
        print("="*80)
        print()
        
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            audio_files.extend(input_dir.glob(ext))
        
        print(f"Found {len(audio_files)} audio files")
        print()
        
        for i, input_path in enumerate(sorted(audio_files), 1):
            print(f"[{i}/{len(audio_files)}] Processing: {input_path.name}")
            output_path = output_dir / f"{input_path.stem}_melody.mp3"
            
            try:
                extract_melody_simple(str(input_path), str(output_path),
                                    percentile=args.percentile, sr=args.sr)
                print()
            except Exception as e:
                print(f"  ✗ Error: {e}")
                print()
                continue
        
        print("="*80)
        print(f"✅ Processed {len(audio_files)} files")
        print(f"Output directory: {output_dir}")
        print("="*80)
        
    else:
        input_path = args.input
        output_path = args.output
        
        if output_path is None:
            base = os.path.splitext(input_path)[0]
            output_path = f"{base}_melody.mp3"
        
        print("="*80)
        print("MELODY EXTRACTION")
        print("="*80)
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Percentile: {args.percentile}")
        print("="*80)
        print()
        
        extract_melody_simple(input_path, output_path,
                            percentile=args.percentile, sr=args.sr)
        
        print()
        print("="*80)
        print("✅ Done!")
        print("="*80)


if __name__ == "__main__":
    main()
