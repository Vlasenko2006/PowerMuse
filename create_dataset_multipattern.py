#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 28 2025

@author: andreyvlasenko

Multi-pattern dataset creation: generates triplets of 3 different songs
for each training example. Each triplet consists of 3 input patterns and
3 corresponding target patterns (next consecutive chunks).
"""

import os
import numpy as np
import random


def create_multipattern_dataset(output_folder="output", dataset_folder="dataset_multipattern", 
                                 chunk_duration=16.33, target_sr=22050, num_patterns=3):
    """
    Creates a multi-pattern dataset from .npy files. For each training example,
    samples 3 different songs and creates triplets of (input, target) chunks.

    Parameters:
    - output_folder: Folder containing source .npy files.
    - dataset_folder: Folder to save the dataset.
    - chunk_duration: Duration of each chunk in seconds (16.33s - architecture requirement).
    - target_sr: Sample rate in Hz (22050 Hz).
    - num_patterns: Number of patterns per triplet (3).
    
    Output format:
    - Each training example: ((input1, input2, input3), (target1, target2, target3))
    - input_i: [2, 360000] stereo audio chunk from song i
    - target_i: [2, 360000] next consecutive chunk from song i
    """
    # Ensure the dataset folder exists
    os.makedirs(dataset_folder, exist_ok=True)

    # List all .npy files in the output folder
    npy_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".npy")])
    
    if len(npy_files) < num_patterns:
        raise ValueError(f"Need at least {num_patterns} songs, found {len(npy_files)}")

    print(f"\n{'='*60}")
    print(f"Creating Multi-Pattern Dataset")
    print(f"{'='*60}")

    # Calculate samples per chunk (must be integer)
    samples_per_chunk = int(chunk_duration * target_sr)
    
    # Process each file and collect all available chunks
    song_chunks = {}  # {song_idx: [(input_chunk, target_chunk), ...]}
    skipped_files = 0
    
    for file_idx, npy_file in enumerate(npy_files):
        file_path = os.path.join(output_folder, npy_file)
        audio_array = np.load(file_path)
        
        # Ensure the audio is long enough for at least one input-target pair
        total_samples = audio_array.shape[-1]
        expected_chunks = 12  # 196s / 16.33s ≈ 12 chunks per song
        
        if total_samples < 2 * samples_per_chunk:  # Need at least 2 chunks for 1 pair
            print(f"Skipping {npy_file}: Too short ({total_samples} samples)")
            skipped_files += 1
            continue
        
        # Split into chunks
        num_full_chunks = min(expected_chunks, total_samples // samples_per_chunk)
        chunks = [
            audio_array[..., i * samples_per_chunk:(i + 1) * samples_per_chunk]
            for i in range(num_full_chunks)
        ]
        
        # Create consecutive pairs (chunk_i -> chunk_i+1)
        pairs = []
        for i in range(len(chunks) - 1):
            input_chunk = chunks[i]
            target_chunk = chunks[i + 1]
            
            # Verify chunk dimensions
            if input_chunk.shape[-1] == samples_per_chunk and target_chunk.shape[-1] == samples_per_chunk:
                pairs.append((input_chunk, target_chunk))
        
        if len(pairs) > 0:
            song_chunks[file_idx] = pairs
            if (file_idx + 1) % 50 == 0:
                print(f"Processed {file_idx + 1}/{len(npy_files)} files...")
    
    # List of song indices with valid chunks
    valid_songs = list(song_chunks.keys())
    
    if len(valid_songs) < num_patterns:
        raise ValueError(f"Only {len(valid_songs)} songs have valid chunks, need {num_patterns}")
    
    print(f"\nValid songs with chunks: {len(valid_songs)}")
    print(f"Generating triplets...")
    
    # Generate triplets by sampling from different songs
    all_triplets = []
    
    # Determine how many triplets to generate
    # Strategy: generate enough to use all available chunks efficiently
    min_pairs_per_song = min(len(pairs) for pairs in song_chunks.values())
    max_triplets = min_pairs_per_song * len(valid_songs) // num_patterns
    
    triplet_count = 0
    max_attempts = max_triplets * 10  # Prevent infinite loops
    attempts = 0
    
    while triplet_count < max_triplets and attempts < max_attempts:
        attempts += 1
        
        # Sample 3 different songs
        sampled_songs = random.sample(valid_songs, num_patterns)
        
        # Sample one chunk pair from each song
        try:
            inputs = []
            targets = []
            
            for song_idx in sampled_songs:
                available_pairs = song_chunks[song_idx]
                if len(available_pairs) == 0:
                    continue
                    
                # Randomly select a pair from this song
                input_chunk, target_chunk = random.choice(available_pairs)
                inputs.append(input_chunk)
                targets.append(target_chunk)
            
            # Only add if we got all 3 patterns
            if len(inputs) == num_patterns and len(targets) == num_patterns:
                triplet = (tuple(inputs), tuple(targets))
                all_triplets.append(triplet)
                triplet_count += 1
                
                if triplet_count % 1000 == 0:
                    print(f"Generated {triplet_count} triplets...")
        
        except Exception as e:
            print(f"Warning: Failed to create triplet: {e}")
            continue
    
    if len(all_triplets) == 0:
        raise ValueError("Failed to generate any valid triplets")
    
    # Shuffle and split into training and validation sets (90/10)
    random.shuffle(all_triplets)
    split_idx = int(0.9 * len(all_triplets))
    training_data = all_triplets[:split_idx]
    validation_data = all_triplets[split_idx:]
    
    # Save the datasets
    training_file = os.path.join(dataset_folder, "training_set_multipattern.npy")
    validation_file = os.path.join(dataset_folder, "validation_set_multipattern.npy")
    
    np.save(training_file, training_data, allow_pickle=True)
    np.save(validation_file, validation_data, allow_pickle=True)
    
    print(f"\n{'='*60}")
    print(f"Multi-Pattern Dataset Created Successfully!")
    print(f"{'='*60}")
    print(f"Total songs processed: {len(valid_songs)}/{len(npy_files)}")
    print(f"Skipped files: {skipped_files}")
    print(f"Training set size: {len(training_data)} triplets")
    print(f"Validation set size: {len(validation_data)} triplets")
    print(f"Patterns per triplet: {num_patterns}")
    print(f"Chunk duration: {chunk_duration:.2f}s")
    print(f"Sample rate: {target_sr} Hz")
    print(f"Samples per chunk: {samples_per_chunk}")
    print(f"Saved to: {dataset_folder}")
    print(f"{'='*60}\n")
    
    # Print example triplet structure
    if len(training_data) > 0:
        example = training_data[0]
        inputs, targets = example
        print(f"Example triplet structure:")
        print(f"  Inputs: {len(inputs)} patterns")
        for i, inp in enumerate(inputs):
            print(f"    Pattern {i+1}: shape {inp.shape}")
        print(f"  Targets: {len(targets)} patterns")
        for i, tgt in enumerate(targets):
            print(f"    Pattern {i+1}: shape {tgt.shape}")
        print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    create_multipattern_dataset(
        output_folder="output",
        dataset_folder="dataset_multipattern",
        chunk_duration=16.33,
        target_sr=22050,
        num_patterns=3
    )
