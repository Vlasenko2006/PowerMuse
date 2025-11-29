#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert .npy validation outputs to audio files (.wav and .mp3)

Usage:
    python convert_outputs_to_audio.py
    
This will convert all .npy files in music_out_multipattern/ to audio files.
"""

import os
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from glob import glob


def npy_to_audio(npy_path, sample_rate=22050):
    """
    Convert .npy file to both .wav and .mp3
    
    Args:
        npy_path: Path to .npy file
        sample_rate: Audio sample rate (default 22050)
    """
    # Load numpy array
    audio = np.load(npy_path)
    
    # Expected shape: [channels, samples] for stereo
    # If mono: [samples], convert to [1, samples]
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    
    # Transpose for soundfile: [samples, channels]
    audio_transposed = audio.T
    
    # Ensure values are in [-1, 1] range
    max_val = np.abs(audio_transposed).max()
    if max_val > 1.0:
        audio_transposed = audio_transposed / max_val
        print(f"  Normalized by {max_val:.4f}")
    
    # Generate output paths
    base_name = os.path.splitext(npy_path)[0]
    wav_path = base_name + '.wav'
    mp3_path = base_name + '.mp3'
    
    # Save as WAV
    sf.write(wav_path, audio_transposed, sample_rate)
    print(f"  ✓ Saved WAV: {wav_path}")
    
    # Convert WAV to MP3
    try:
        audio_segment = AudioSegment.from_wav(wav_path)
        audio_segment.export(mp3_path, format='mp3', bitrate='320k')
        print(f"  ✓ Saved MP3: {mp3_path}")
    except Exception as e:
        print(f"  ✗ MP3 conversion failed: {e}")
    
    # Print audio info
    duration = audio.shape[1] / sample_rate
    print(f"  Duration: {duration:.2f}s, Channels: {audio.shape[0]}, Shape: {audio.shape}")


def convert_all_outputs(music_out_folder="music_out_multipattern", sample_rate=22050):
    """
    Convert all .npy files in the music output folder to audio.
    
    Args:
        music_out_folder: Folder containing .npy files
        sample_rate: Audio sample rate
    """
    if not os.path.exists(music_out_folder):
        print(f"Error: Folder '{music_out_folder}' does not exist.")
        return
    
    # Find all .npy files
    npy_files = glob(os.path.join(music_out_folder, "*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in '{music_out_folder}'")
        return
    
    print(f"\n{'='*60}")
    print(f"Converting {len(npy_files)} .npy files to audio")
    print(f"{'='*60}\n")
    
    converted_count = 0
    for npy_path in sorted(npy_files):
        print(f"Processing: {os.path.basename(npy_path)}")
        try:
            npy_to_audio(npy_path, sample_rate)
            converted_count += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
        print()
    
    print(f"{'='*60}")
    print(f"Conversion complete!")
    print(f"Successfully converted: {converted_count}/{len(npy_files)} files")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert .npy validation outputs to audio files")
    parser.add_argument(
        "--folder",
        type=str,
        default="music_out_multipattern",
        help="Folder containing .npy files (default: music_out_multipattern)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Audio sample rate (default: 22050)"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Convert single .npy file instead of entire folder"
    )
    
    args = parser.parse_args()
    
    if args.file:
        # Convert single file
        print(f"Converting single file: {args.file}")
        npy_to_audio(args.file, args.sample_rate)
    else:
        # Convert entire folder
        convert_all_outputs(args.folder, args.sample_rate)
