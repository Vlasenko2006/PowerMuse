#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract rhythm (percussion/drums) from audio using harmonic-percussive source separation.

Uses librosa's HPSS (Harmonic-Percussive Source Separation) to isolate rhythmic components.
"""

import os
import argparse
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


def extract_rhythm(audio_path, output_path, margin=8.0, sr=24000):
    """
    Extract rhythm/percussion from audio using HPSS.
    
    Args:
        audio_path: Input audio file path
        output_path: Output file path (.mp3, .wav, etc.)
        margin: Separation margin (higher = more separation, 1-16 typical)
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
    
    # Convert percussive component back to time domain
    y_percussive = librosa.istft(P)
    
    # Normalize to prevent clipping
    y_percussive = y_percussive / (np.abs(y_percussive).max() + 1e-8)
    y_percussive = np.clip(y_percussive, -0.99, 0.99)
    
    # Compute energy ratio
    orig_energy = np.sqrt(np.mean(y ** 2))
    rhythm_energy = np.sqrt(np.mean(y_percussive ** 2))
    ratio = rhythm_energy / (orig_energy + 1e-8)
    
    print(f"  Original RMS: {orig_energy:.4f}")
    print(f"  Rhythm RMS: {rhythm_energy:.4f}")
    print(f"  Energy ratio: {ratio:.2%}")
    
    # Save output
    if output_path.endswith('.mp3'):
        # For MP3, use soundfile with appropriate subtype
        print(f"  Saving MP3: {output_path}")
        # Note: soundfile doesn't support MP3 directly, use pydub as fallback
        try:
            from pydub import AudioSegment
            # Save as WAV first, then convert
            temp_wav = output_path.replace('.mp3', '_temp.wav')
            sf.write(temp_wav, y_percussive, sr)
            # Convert to MP3
            audio = AudioSegment.from_wav(temp_wav)
            audio.export(output_path, format='mp3', bitrate='192k')
            os.remove(temp_wav)
            print(f"  ✓ Saved: {output_path}")
        except ImportError:
            print("  ⚠️  pydub not installed, saving as WAV instead")
            output_path = output_path.replace('.mp3', '.wav')
            sf.write(output_path, y_percussive, sr)
            print(f"  ✓ Saved: {output_path}")
    else:
        # WAV or other format
        sf.write(output_path, y_percussive, sr)
        print(f"  ✓ Saved: {output_path}")
    
    return y_percussive


def main():
    parser = argparse.ArgumentParser(description='Extract rhythm from audio files')
    parser.add_argument('--input', type=str, required=True,
                       help='Input audio file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: input_rhythm.mp3)')
    parser.add_argument('--margin', type=float, default=8.0,
                       help='HPSS separation margin (1-16, higher=more separation)')
    parser.add_argument('--sr', type=int, default=24000,
                       help='Sample rate')
    parser.add_argument('--batch', action='store_true',
                       help='Process all files in input directory')
    parser.add_argument('--output_dir', type=str, default='rhythm_extracted',
                       help='Output directory for batch mode')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch mode: process all audio files in directory
        input_dir = Path(args.input)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("="*80)
        print("BATCH RHYTHM EXTRACTION")
        print("="*80)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Margin: {args.margin}")
        print("="*80)
        print()
        
        # Find all audio files
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            audio_files.extend(input_dir.glob(ext))
        
        print(f"Found {len(audio_files)} audio files")
        print()
        
        for i, input_path in enumerate(sorted(audio_files), 1):
            print(f"[{i}/{len(audio_files)}] Processing: {input_path.name}")
            output_path = output_dir / f"{input_path.stem}_rhythm.mp3"
            
            try:
                extract_rhythm(str(input_path), str(output_path), 
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
        # Single file mode
        input_path = args.input
        output_path = args.output
        
        if output_path is None:
            # Auto-generate output path
            base = os.path.splitext(input_path)[0]
            output_path = f"{base}_rhythm.mp3"
        
        print("="*80)
        print("RHYTHM EXTRACTION")
        print("="*80)
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Margin: {args.margin}")
        print("="*80)
        print()
        
        extract_rhythm(input_path, output_path, margin=args.margin, sr=args.sr)
        
        print()
        print("="*80)
        print("✅ Done!")
        print("="*80)


if __name__ == "__main__":
    main()
