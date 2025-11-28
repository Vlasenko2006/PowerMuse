#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 13:23:09 2025

@author: andreyvlasenko

Enhanced conversion script for neural network outputs with proper normalization.
"""

import numpy as np
from numpy_2_mp3 import numpy_to_mp3
import os


# Configuration
sample_rate = 22050  # Updated from 12000 Hz
path = "/Users/andreyvlasenko/tst/Music_NN/Lets_Rock/NN_output/"
path_output = "/Users/andreyvlasenko/tst/Music_NN/Lets_Rock/NN_music_output"

# Alternative paths for enhanced model outputs
path_enhanced = "music_out_enhanced"
path_output_enhanced = "music_mp3_enhanced"

# Ensure the output directory exists
os.makedirs(path_output, exist_ok=True)
os.makedirs(path_output_enhanced, exist_ok=True)

print("="*60)
print("NEURAL NETWORK OUTPUT TO MP3 CONVERTER")
print("="*60)
print(f"Sample rate: {sample_rate} Hz")
print(f"Input path: {path}")
print(f"Output path: {path_output}")
print("="*60)


def convert_numpy_to_mp3(input_path, output_path, sample_rate=22050, remove_offset=False):
    """
    Convert .npy files to .mp3 files with proper normalization.
    
    Parameters:
    - input_path: Folder containing .npy files
    - output_path: Folder to save .mp3 files
    - sample_rate: Audio sample rate
    - remove_offset: If True, subtract 1 from array (for old normalization method)
    """
    converted_count = 0
    
    # Process all .npy files
    for file in os.listdir(input_path):
        if file.endswith(".npy"):
            file_path = os.path.join(input_path, file)
            print(f"\nProcessing: {file}")
            
            try:
                # Load the NumPy array
                array = np.load(file_path)
                
                # Remove offset if using old normalization (inputs + 1)
                if remove_offset:
                    array = array - 1.0
                
                # Data is already in [-1, 1] range from new normalization
                # Just convert to float32 if needed
                array = array.astype(np.float32)
                
                # Clip values to ensure they're in valid range
                array = np.clip(array, -1.0, 1.0)
                
                print(f"  Shape: {array.shape}")
                print(f"  Min: {array.min():.4f}, Max: {array.max():.4f}")
                
                # Generate output MP3 file path
                output_mp3_file = os.path.join(output_path, file.replace(".npy", ".mp3"))
                
                # Convert to MP3
                numpy_to_mp3(array, sample_rate, output_mp3_file)
                converted_count += 1
                
            except Exception as e:
                print(f"  ERROR: {e}")
    
    return converted_count


# Convert files from default path
print("\nConverting files from default path...")
count1 = convert_numpy_to_mp3(path, path_output, sample_rate, remove_offset=True)

# Convert files from enhanced model path if it exists
if os.path.exists(path_enhanced):
    print("\n\nConverting files from enhanced model path...")
    count2 = convert_numpy_to_mp3(path_enhanced, path_output_enhanced, sample_rate, remove_offset=False)
else:
    count2 = 0
    print(f"\nEnhanced model path '{path_enhanced}' not found, skipping.")

print("\n" + "="*60)
print("CONVERSION COMPLETE!")
print("="*60)
print(f"Total files converted: {count1 + count2}")
print(f"Default path: {count1} files -> {path_output}")
if count2 > 0:
    print(f"Enhanced path: {count2} files -> {path_output_enhanced}")
print("="*60)
