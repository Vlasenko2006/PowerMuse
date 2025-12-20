"""
Dataset loader for 24-second audio pairs (WAV format)
For use with Adaptive Window Selection
"""

import os
import torch
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class AudioPairsDataset24sec(Dataset):
    """
    Dataset for loading 24-second audio pairs in WAV format.
    
    Each sample is a pair:
        - input: 24 seconds of audio
        - target: 24 seconds of audio (continuation)
    
    Compatible with adaptive window selection (agent selects 3×16sec windows).
    """
    
    def __init__(self, data_folder, split='train', target_sr=24000):
        """
        Args:
            data_folder: Root folder containing train/ and val/ subfolders
            split: 'train' or 'val'
            target_sr: Target sample rate (default 24000 Hz)
        """
        self.data_folder = Path(data_folder) / split
        self.target_sr = target_sr
        self.split = split
        
        # Find all input files
        self.input_files = sorted(self.data_folder.glob('pair_*_input.wav'))
        
        if len(self.input_files) == 0:
            raise ValueError(f"No audio files found in {self.data_folder}")
        
        print(f"AudioPairsDataset24sec ({split}):")
        print(f"  Folder: {self.data_folder}")
        print(f"  Number of pairs: {len(self.input_files)}")
        print(f"  Sample rate: {target_sr} Hz")
        print(f"  Duration: 24 seconds per file")
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_audio: [1, samples] - mono audio (24 seconds)
            target_audio: [1, samples] - mono audio (24 seconds)
        """
        input_path = self.input_files[idx]
        output_path = input_path.parent / input_path.name.replace('_input.wav', '_output.wav')
        
        # Load audio files
        try:
            audio_input, sr_input = sf.read(input_path)
            audio_target, sr_target = sf.read(output_path)
        except Exception as e:
            print(f"Error loading {input_path}: {e}")
            # Return zeros on error
            return torch.zeros(1, int(24.0 * self.target_sr)), torch.zeros(1, int(24.0 * self.target_sr))
        
        # Convert stereo to mono if needed
        if len(audio_input.shape) == 2:
            audio_input = audio_input.mean(axis=1)
        if len(audio_target.shape) == 2:
            audio_target = audio_target.mean(axis=1)
        
        # Resample if needed (though should already be correct)
        if sr_input != self.target_sr:
            print(f"Warning: {input_path} has sr={sr_input}, expected {self.target_sr}")
        
        # Convert to torch tensors [1, samples]
        audio_input = torch.from_numpy(audio_input).float().unsqueeze(0)
        audio_target = torch.from_numpy(audio_target).float().unsqueeze(0)
        
        # Normalize to [-1, 1]
        max_val = max(audio_input.abs().max(), audio_target.abs().max())
        if max_val > 1.0:
            audio_input = audio_input / max_val
            audio_target = audio_target / max_val
        
        return audio_input, audio_target


def collate_fn(batch):
    """
    Custom collate function to stack audio pairs into batches.
    
    Args:
        batch: List of (input, target) tuples
    
    Returns:
        input_batch: [B, 1, samples]
        target_batch: [B, 1, samples]
    """
    inputs, targets = zip(*batch)
    
    # Stack into batches
    input_batch = torch.stack(inputs, dim=0)  # [B, 1, samples]
    target_batch = torch.stack(targets, dim=0)  # [B, 1, samples]
    
    return input_batch, target_batch


if __name__ == "__main__":
    """Test the dataset loader"""
    print("Testing AudioPairsDataset24sec...")
    
    # Create dataset
    dataset = AudioPairsDataset24sec(
        data_folder='dataset_pairs_wav_24sec',
        split='train',
        target_sr=24000
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Test loading one sample
    input_audio, target_audio = dataset[0]
    print(f"\nSample 0:")
    print(f"  Input shape: {input_audio.shape}")
    print(f"  Target shape: {target_audio.shape}")
    print(f"  Input range: [{input_audio.min():.3f}, {input_audio.max():.3f}]")
    print(f"  Target range: [{target_audio.min():.3f}, {target_audio.max():.3f}]")
    print(f"  Duration: {input_audio.shape[1] / 24000:.2f} seconds")
    
    # Test DataLoader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"\nTesting DataLoader (batch_size=4)...")
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f"  Batch {batch_idx}:")
        print(f"    Inputs: {inputs.shape}")
        print(f"    Targets: {targets.shape}")
        if batch_idx >= 2:
            break
    
    print("\n✓ Dataset loader test passed!")
