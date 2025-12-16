#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract treble (high-frequency content) from audio.

Uses high-pass filtering to isolate high-frequency components (cymbals, hi-hats, air, brightness).
"""

import os
import argparse
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy import signal


def extract_treble(audio_path, output_path, cutoff_freq=2000, order=6, sr=24000):
    """
    Extract treble (high frequencies) using high-pass filter.
    
    Args:
        audio_path: Input audio file path
        output_path: Output file path
        cutoff_freq: High-pass filter cutoff frequency in Hz (1000-8000 typical)
        order: Filter order (4-8 typical, higher = steeper)
        sr: Target sample rate
    """
    print(f"Loading: {audio_path}")
    
    # Load audio
    y, orig_sr = librosa.load(audio_path, sr=sr, mono=True)
    print(f"  Duration: {len(y)/sr:.2f}s, SR: {sr} Hz")
    
    # Design high-pass Butterworth filter
    print(f"  Applying high-pass filter (cutoff={cutoff_freq}Hz, order={order})...")
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist
    
    # Ensure cutoff is valid
    normalized_cutoff = np.clip(normalized_cutoff, 0.01, 0.99)
    
    # Design filter
    sos = signal.butter(order, normalized_cutoff, btype='high', output='sos')
    
    # Apply filter (using sosfiltfilt for zero-phase filtering)
    y_treble = signal.sosfiltfilt(sos, y)
    
    # Normalize to prevent clipping
    y_treble = y_treble / (np.abs(y_treble).max() + 1e-8)
    y_treble = np.clip(y_treble, -0.99, 0.99)
    
    # Compute energy ratio
    orig_energy = np.sqrt(np.mean(y ** 2))
    treble_energy = np.sqrt(np.mean(y_treble ** 2))
    ratio = treble_energy / (orig_energy + 1e-8)
    
    # Compute spectral centroid (brightness measure)
    centroid_orig = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    centroid_treble = librosa.feature.spectral_centroid(y=y_treble, sr=sr)[0].mean()
    
    print(f"  Original RMS: {orig_energy:.4f}")
    print(f"  Treble RMS: {treble_energy:.4f}")
    print(f"  Energy ratio: {ratio:.2%}")
    print(f"  Spectral centroid: {centroid_orig:.0f}Hz → {centroid_treble:.0f}Hz")
    
    # Save output
    save_audio(y_treble, output_path, sr)
    
    return y_treble


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
    parser = argparse.ArgumentParser(description='Extract treble from audio files')
    parser.add_argument('--input', type=str, required=True,
                       help='Input audio file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: input_treble.mp3)')
    parser.add_argument('--cutoff', type=float, default=2000,
                       help='High-pass filter cutoff frequency in Hz (1000-8000)')
    parser.add_argument('--order', type=int, default=6,
                       help='Filter order (4-8, higher=steeper)')
    parser.add_argument('--sr', type=int, default=24000,
                       help='Sample rate')
    parser.add_argument('--batch', action='store_true',
                       help='Process all files in input directory')
    parser.add_argument('--output_dir', type=str, default='treble_extracted',
                       help='Output directory for batch mode')
    
    args = parser.parse_args()
    
    if args.batch:
        input_dir = Path(args.input)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("="*80)
        print("BATCH TREBLE EXTRACTION")
        print("="*80)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Cutoff: {args.cutoff}Hz, Order: {args.order}")
        print("="*80)
        print()
        
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            audio_files.extend(input_dir.glob(ext))
        
        print(f"Found {len(audio_files)} audio files")
        print()
        
        for i, input_path in enumerate(sorted(audio_files), 1):
            print(f"[{i}/{len(audio_files)}] Processing: {input_path.name}")
            output_path = output_dir / f"{input_path.stem}_treble.mp3"
            
            try:
                extract_treble(str(input_path), str(output_path),
                             cutoff_freq=args.cutoff, order=args.order, sr=args.sr)
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
            output_path = f"{base}_treble.mp3"
        
        print("="*80)
        print("TREBLE EXTRACTION")
        print("="*80)
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Cutoff: {args.cutoff}Hz, Order: {args.order}")
        print("="*80)
        print()
        
        extract_treble(input_path, output_path,
                      cutoff_freq=args.cutoff, order=args.order, sr=args.sr)
        
        print()
        print("="*80)
        print("✅ Done!")
        print("="*80)


if __name__ == "__main__":
    main()
