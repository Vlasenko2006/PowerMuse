#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract harmony (harmonic/tonal content) from audio.

Uses HPSS to isolate harmonic components (chords, bass, sustained notes).
"""

import os
import argparse
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


def extract_harmony(audio_path, output_path, margin=2.0, sr=24000):
    """
    Extract harmonic content (chords, sustained notes) from audio using HPSS.
    
    Args:
        audio_path: Input audio file path
        output_path: Output file path
        margin: Separation margin (lower = keep more harmonic, 1-4 typical)
        sr: Target sample rate
    """
    print(f"Loading: {audio_path}")
    
    # Load audio
    y, orig_sr = librosa.load(audio_path, sr=sr, mono=True)
    print(f"  Duration: {len(y)/sr:.2f}s, SR: {sr} Hz")
    
    # Compute STFT
    D = librosa.stft(y)
    
    # Separate harmonic and percussive components
    print(f"  Separating with margin={margin}...")
    H, P = librosa.decompose.hpss(D, margin=margin)
    
    # Convert harmonic component back to time domain
    y_harmonic = librosa.istft(H)
    
    # Normalize to prevent clipping
    y_harmonic = y_harmonic / (np.abs(y_harmonic).max() + 1e-8)
    y_harmonic = np.clip(y_harmonic, -0.99, 0.99)
    
    # Compute energy ratio
    orig_energy = np.sqrt(np.mean(y ** 2))
    harmony_energy = np.sqrt(np.mean(y_harmonic ** 2))
    ratio = harmony_energy / (orig_energy + 1e-8)
    
    print(f"  Original RMS: {orig_energy:.4f}")
    print(f"  Harmony RMS: {harmony_energy:.4f}")
    print(f"  Energy ratio: {ratio:.2%}")
    
    # Save output
    save_audio(y_harmonic, output_path, sr)
    
    return y_harmonic


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
    parser = argparse.ArgumentParser(description='Extract harmony from audio files')
    parser.add_argument('--input', type=str, required=True,
                       help='Input audio file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: input_harmony.mp3)')
    parser.add_argument('--margin', type=float, default=2.0,
                       help='HPSS separation margin (1-4, lower=more harmonic)')
    parser.add_argument('--sr', type=int, default=24000,
                       help='Sample rate')
    parser.add_argument('--batch', action='store_true',
                       help='Process all files in input directory')
    parser.add_argument('--output_dir', type=str, default='harmony_extracted',
                       help='Output directory for batch mode')
    
    args = parser.parse_args()
    
    if args.batch:
        input_dir = Path(args.input)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("="*80)
        print("BATCH HARMONY EXTRACTION")
        print("="*80)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Margin: {args.margin}")
        print("="*80)
        print()
        
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            audio_files.extend(input_dir.glob(ext))
        
        print(f"Found {len(audio_files)} audio files")
        print()
        
        for i, input_path in enumerate(sorted(audio_files), 1):
            print(f"[{i}/{len(audio_files)}] Processing: {input_path.name}")
            output_path = output_dir / f"{input_path.stem}_harmony.mp3"
            
            try:
                extract_harmony(str(input_path), str(output_path),
                              margin=args.margin, sr=args.sr)
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
            output_path = f"{base}_harmony.mp3"
        
        print("="*80)
        print("HARMONY EXTRACTION")
        print("="*80)
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Margin: {args.margin}")
        print("="*80)
        print()
        
        extract_harmony(input_path, output_path,
                       margin=args.margin, sr=args.sr)
        
        print()
        print("="*80)
        print("✅ Done!")
        print("="*80)


if __name__ == "__main__":
    main()
