#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-efficient multi-pattern dataset creation
Generates triplets incrementally without loading everything into memory
"""

import os
import numpy as np
import random


def create_multipattern_dataset(output_folder="output", dataset_folder="dataset_multipattern",
                                chunk_duration=16.33, target_sr=22050, num_patterns=3,
                                max_triplets=10000):
    """
    Memory-efficient dataset creation: processes files incrementally
    """
    os.makedirs(dataset_folder, exist_ok=True)

    npy_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".npy")])
    
    if len(npy_files) < num_patterns:
        raise ValueError(f"Need at least {num_patterns} songs, found {len(npy_files)}")

    print(f"\n{'='*60}")
    print(f"Creating Multi-Pattern Dataset (Memory-Efficient)")
    print(f"{'='*60}")

    samples_per_chunk = int(chunk_duration * target_sr)
    
    # First pass: collect metadata about available chunks per song
    print("Phase 1: Scanning audio files...")
    song_metadata = {}  # {song_idx: num_pairs}
    
    for file_idx, npy_file in enumerate(npy_files):
        file_path = os.path.join(output_folder, npy_file)
        
        try:
            # Load only shape info, not full data
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
    
    # Calculate target number of triplets
    min_pairs = min(meta['num_pairs'] for meta in song_metadata.values())
    target_triplets = min(max_triplets, min_pairs * len(valid_songs) // num_patterns)
    
    print(f"Phase 2: Generating {target_triplets} triplets...")
    
    # Generate triplets incrementally and save in batches
    training_file = os.path.join(dataset_folder, "training_set_multipattern.npy")
    validation_file = os.path.join(dataset_folder, "validation_set_multipattern.npy")
    
    training_data = []
    validation_data = []
    batch_size = 100  # Process in small batches
    
    generated = 0
    attempts = 0
    max_attempts = target_triplets * 20
    
    # Cache for loaded chunks to avoid repeated disk access
    chunk_cache = {}
    cache_size_limit = 50  # Keep max 50 chunks in memory
    
    def load_chunk_pair(song_idx, pair_idx):
        """Load a specific chunk pair from disk"""
        cache_key = (song_idx, pair_idx)
        if cache_key in chunk_cache:
            return chunk_cache[cache_key]
        
        file_path = os.path.join(output_folder, song_metadata[song_idx]['file'])
        audio_array = np.load(file_path)
        
        # Extract the specific pair
        input_chunk = audio_array[..., pair_idx * samples_per_chunk:(pair_idx + 1) * samples_per_chunk]
        target_chunk = audio_array[..., (pair_idx + 1) * samples_per_chunk:(pair_idx + 2) * samples_per_chunk]
        
        # Validate
        if input_chunk.shape[-1] != samples_per_chunk or target_chunk.shape[-1] != samples_per_chunk:
            return None, None
        
        # Cache if space available
        if len(chunk_cache) < cache_size_limit:
            chunk_cache[cache_key] = (input_chunk, target_chunk)
        
        return input_chunk, target_chunk
    
    while generated < target_triplets and attempts < max_attempts:
        attempts += 1
        
        # Sample 3 different songs
        try:
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
                triplet = (tuple(inputs), tuple(targets))
                
                # 90/10 split
                if random.random() < 0.9:
                    training_data.append(triplet)
                else:
                    validation_data.append(triplet)
                
                generated += 1
                
                # Save in batches to free memory
                if generated % batch_size == 0:
                    print(f"Generated {generated}/{target_triplets} triplets...")
                    chunk_cache.clear()  # Clear cache periodically
        
        except Exception as e:
            continue
    
    if len(training_data) == 0:
        raise ValueError("Failed to generate any triplets")
    
    # Final save
    print("Saving datasets...")
    np.save(training_file, training_data, allow_pickle=True)
    np.save(validation_file, validation_data, allow_pickle=True)
    
    print(f"\n{'='*60}")
    print(f"Dataset Created Successfully!")
    print(f"{'='*60}")
    print(f"Training set: {len(training_data)} triplets")
    print(f"Validation set: {len(validation_data)} triplets")
    print(f"Saved to: {dataset_folder}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    create_multipattern_dataset(
        output_folder="output",
        dataset_folder="dataset_multipattern",
        chunk_duration=16.33,
        target_sr=22050,
        num_patterns=3,
        max_triplets=5000  # Limit to 5000 triplets to save memory
    )
