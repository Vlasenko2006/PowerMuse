#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 12:12:20 2025

@author: andreyvlasenko

Enhanced dataset creation with 16-second chunks and improved sample rate.
"""

import os
import numpy as np
import random

def create_dataset(output_folder="output", dataset_folder="dataset", chunk_duration=16, target_sr=22050):
    """
    Creates a dataset from .npy files. Splits each file into chunks of equal duration,
    combines odd and even chunks into input-target pairs, scrambles them, and splits
    into training and validation sets.

    Parameters:
    - output_folder: Folder containing source .npy files.
    - dataset_folder: Folder to save the dataset.
    - chunk_duration: Duration of each chunk in seconds (16s for better quality).
    - target_sr: Sample rate in Hz (22050 Hz for better quality).
    """
    # Ensure the dataset folder exists
    os.makedirs(dataset_folder, exist_ok=True)

    # List all .npy files in the output folder
    npy_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".npy")])

    # Initialize containers for training and validation data
    training_data = []
    validation_data = []

    # Process each .npy file
    chunk_count = 0  # Counter for chunks
    skipped_files = 0
    
    for file_idx, npy_file in enumerate(npy_files, start=1):
        # Load the .npy file
        file_path = os.path.join(output_folder, npy_file)
        audio_array = np.load(file_path)

        # Calculate the number of samples in a chunk (16s at 22050 Hz = 352,800 samples)
        samples_per_chunk = chunk_duration * target_sr

        # Ensure the audio is long enough and can be split into 12 chunks (192s total)
        total_samples = audio_array.shape[-1]
        expected_chunks = 12  # 192s / 16s = 12 chunks
        
        if total_samples < expected_chunks * samples_per_chunk:
            print(f"Skipping {npy_file}: Not enough samples to create {expected_chunks} "
                  f"chunks of {chunk_duration}s. "
                  f"Need {expected_chunks * samples_per_chunk}, got {total_samples}")
            skipped_files += 1
            continue

        # Split the audio into 12 chunks of 16 seconds each
        chunks = [
            audio_array[..., i * samples_per_chunk:(i + 1) * samples_per_chunk]
            for i in range(expected_chunks)
        ]

        # Verify chunk dimensions
        for i, chunk in enumerate(chunks):
            if chunk.shape[-1] != samples_per_chunk:
                print(f"Warning: Chunk {i} in {npy_file} has incorrect size: {chunk.shape}")

        # Combine consecutive chunks into input-target pairs (chunks 0->1, 2->3, etc.)
        pairs = [
            (chunks[i], chunks[i + 1]) for i in range(0, len(chunks) - 1, 2)
        ]
        chunk_count += len(pairs)

        # Add to the combined list
        training_data.extend(pairs)
        
        if (file_idx % 50) == 0:
            print(f"Processed {file_idx}/{len(npy_files)} files, created {chunk_count} pairs so far...")

    # Split into training and validation sets (90/10 split)
    random.shuffle(training_data)  # Shuffle the entire dataset
    split_idx = int(0.9 * len(training_data))  # 90% for training
    validation_data = training_data[split_idx:]
    training_data = training_data[:split_idx]

    # Save the training and validation sets
    training_file = os.path.join(dataset_folder, "training_set.npy")
    validation_file = os.path.join(dataset_folder, "validation_set.npy")
    np.save(training_file, training_data)
    np.save(validation_file, validation_data)

    print(f"\n{'='*60}")
    print(f"Dataset created successfully!")
    print(f"{'='*60}")
    print(f"Processed files: {len(npy_files) - skipped_files}/{len(npy_files)}")
    print(f"Skipped files: {skipped_files}")
    print(f"Training set size: {len(training_data)} pairs")
    print(f"Validation set size: {len(validation_data)} pairs")
    print(f"Chunk duration: {chunk_duration}s")
    print(f"Sample rate: {target_sr} Hz")
    print(f"Samples per chunk: {chunk_duration * target_sr}")
    print(f"Saved to: {dataset_folder}")
    print(f"{'='*60}")


# Example usage
if __name__ == "__main__":
    create_dataset(
        output_folder="output", 
        dataset_folder="dataset", 
        chunk_duration=16,  # 16 seconds per chunk
        target_sr=22050     # Higher quality sample rate
    )
