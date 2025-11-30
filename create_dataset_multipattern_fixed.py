#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-efficient multi-pattern dataset creation with proper format
Saves each triplet as a separate file for true memory-mapped access
"""

import os
import numpy as np
import random


def create_multipattern_dataset(output_folder="output", dataset_folder="dataset_multipattern",
                                chunk_duration=16.33, target_sr=22050, num_patterns=3,
                                max_triplets=2000):
    """
    Creates dataset with memory-mapped friendly format.
    Saves as standard numpy arrays instead of pickled objects.
    """
    os.makedirs(dataset_folder, exist_ok=True)

    npy_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".npy")])
    
    if len(npy_files) < num_patterns:
        raise ValueError(f"Need at least {num_patterns} songs, found {len(npy_files)}")

    print(f"\n{'='*60}")
    print(f"Creating Multi-Pattern Dataset")
    print(f"{'='*60}")

    samples_per_chunk = int(chunk_duration * target_sr)
    
    # Phase 1: Scan files
    print("Phase 1: Scanning audio files...")
    song_metadata = {}
    
    for file_idx, npy_file in enumerate(npy_files):
        file_path = os.path.join(output_folder, npy_file)
        
        try:
            audio_array = np.load(file_path, mmap_mode='r')
            
            if audio_array.ndim != 2 or audio_array.shape[0] != 2:
                continue
            
            total_samples = audio_array.shape[-1]
            if total_samples < 2 * samples_per_chunk:
                continue
            
            num_chunks = min(12, total_samples // samples_per_chunk)
            num_pairs = max(0, num_chunks - 1)
            
            if num_pairs > 0:
                song_metadata[file_idx] = {
                    'file': npy_file,
                    'num_pairs': num_pairs
                }
                
        except Exception as e:
            print(f"Skipping {npy_file}: {e}")
            continue
    
    valid_songs = list(song_metadata.keys())
    if len(valid_songs) < num_patterns:
        raise ValueError(f"Only {len(valid_songs)} valid songs, need {num_patterns}")
    
    print(f"Found {len(valid_songs)} valid songs")
    
    # Calculate target triplets
    min_pairs = min(meta['num_pairs'] for meta in song_metadata.values())
    target_triplets = min(max_triplets, min_pairs * len(valid_songs) // num_patterns)
    
    print(f"Phase 2: Generating {target_triplets} triplets...")
    
    # Pre-allocate arrays for better memory efficiency
    # Shape: [num_triplets, num_patterns, 2, 360000]
    training_size = int(target_triplets * 0.9)
    validation_size = target_triplets - training_size
    
    train_inputs = np.zeros((training_size, num_patterns, 2, samples_per_chunk), dtype=np.float32)
    train_targets = np.zeros((training_size, num_patterns, 2, samples_per_chunk), dtype=np.float32)
    val_inputs = np.zeros((validation_size, num_patterns, 2, samples_per_chunk), dtype=np.float32)
    val_targets = np.zeros((validation_size, num_patterns, 2, samples_per_chunk), dtype=np.float32)
    
    def load_chunk_pair(song_idx, pair_idx):
        """Load specific chunk pair from disk"""
        file_path = os.path.join(output_folder, song_metadata[song_idx]['file'])
        audio_array = np.load(file_path)
        
        input_chunk = audio_array[..., pair_idx * samples_per_chunk:(pair_idx + 1) * samples_per_chunk]
        target_chunk = audio_array[..., (pair_idx + 1) * samples_per_chunk:(pair_idx + 2) * samples_per_chunk]
        
        if input_chunk.shape[-1] != samples_per_chunk or target_chunk.shape[-1] != samples_per_chunk:
            return None, None
        
        return input_chunk, target_chunk
    
    train_idx = 0
    val_idx = 0
    attempts = 0
    max_attempts = target_triplets * 20
    
    while (train_idx + val_idx) < target_triplets and attempts < max_attempts:
        attempts += 1
        
        try:
            # Sample 3 different songs
            sampled_songs = random.sample(valid_songs, num_patterns)
            
            inputs = []
            targets = []
            valid = True
            
            for song_idx in sampled_songs:
                max_pairs = song_metadata[song_idx]['num_pairs']
                pair_idx = random.randint(0, max_pairs - 1)
                
                input_chunk, target_chunk = load_chunk_pair(song_idx, pair_idx)
                
                if input_chunk is None:
                    valid = False
                    break
                
                inputs.append(input_chunk)
                targets.append(target_chunk)
            
            if valid and len(inputs) == num_patterns:
                # 90/10 split
                if random.random() < 0.9 and train_idx < training_size:
                    for i in range(num_patterns):
                        train_inputs[train_idx, i] = inputs[i]
                        train_targets[train_idx, i] = targets[i]
                    train_idx += 1
                elif val_idx < validation_size:
                    for i in range(num_patterns):
                        val_inputs[val_idx, i] = inputs[i]
                        val_targets[val_idx, i] = targets[i]
                    val_idx += 1
                
                if (train_idx + val_idx) % 100 == 0:
                    print(f"Generated {train_idx + val_idx}/{target_triplets} triplets...")
        
        except Exception as e:
            continue
    
    # Trim to actual size
    train_inputs = train_inputs[:train_idx]
    train_targets = train_targets[:train_idx]
    val_inputs = val_inputs[:val_idx]
    val_targets = val_targets[:val_idx]
    
    # Save as standard numpy arrays (no pickle, memory-map compatible)
    print("Saving datasets...")
    np.save(os.path.join(dataset_folder, "train_inputs.npy"), train_inputs)
    np.save(os.path.join(dataset_folder, "train_targets.npy"), train_targets)
    np.save(os.path.join(dataset_folder, "val_inputs.npy"), val_inputs)
    np.save(os.path.join(dataset_folder, "val_targets.npy"), val_targets)
    
    print(f"\n{'='*60}")
    print(f"Dataset Created Successfully!")
    print(f"{'='*60}")
    print(f"Training triplets: {train_idx}")
    print(f"Validation triplets: {val_idx}")
    print(f"Input shape: {train_inputs.shape}")
    print(f"Target shape: {train_targets.shape}")
    print(f"Saved to: {dataset_folder}/")
    print(f"Files: train_inputs.npy, train_targets.npy, val_inputs.npy, val_targets.npy")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    create_multipattern_dataset(
        output_folder="output",
        dataset_folder="dataset_multipattern",
        chunk_duration=16.33,
        target_sr=22050,
        num_patterns=3,
        max_triplets=2000
    )
