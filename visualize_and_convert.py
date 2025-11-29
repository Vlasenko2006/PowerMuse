#!/usr/bin/env python3
"""
Visualize multi-pattern fusion results and convert to audio files.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf
import os

# Configuration
SAMPLE_RATE = 22050
EPOCH = 50
INPUT_DIR = 'mus_output'
OUTPUT_DIR = 'audio_output'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("MULTI-PATTERN FUSION - RESULTS ANALYSIS")
print("=" * 60)

# Load all validation outputs
print("\nLoading validation samples...")
input1 = np.load(f'{INPUT_DIR}/input_pattern1_epoch_{EPOCH}.npy')
input2 = np.load(f'{INPUT_DIR}/input_pattern2_epoch_{EPOCH}.npy')
input3 = np.load(f'{INPUT_DIR}/input_pattern3_epoch_{EPOCH}.npy')
recon1 = np.load(f'{INPUT_DIR}/reconstructed_pattern1_epoch_{EPOCH}.npy')
recon2 = np.load(f'{INPUT_DIR}/reconstructed_pattern2_epoch_{EPOCH}.npy')
recon3 = np.load(f'{INPUT_DIR}/reconstructed_pattern3_epoch_{EPOCH}.npy')
fused = np.load(f'{INPUT_DIR}/output_fused_epoch_{EPOCH}.npy')
target = np.load(f'{INPUT_DIR}/target_reference_epoch_{EPOCH}.npy')

print(f"✓ All files loaded successfully")
print(f"  Duration: {fused.shape[1] / SAMPLE_RATE:.2f} seconds")
print(f"  Channels: {fused.shape[0]} (stereo)")
print(f"  Sample rate: {SAMPLE_RATE} Hz")

# ============================================================
# 1. GENERATE AUDIO FILES
# ============================================================
print("\n" + "=" * 60)
print("GENERATING AUDIO FILES")
print("=" * 60)

audio_files = {
    'input_pattern1': input1,
    'input_pattern2': input2,
    'input_pattern3': input3,
    'reconstructed_pattern1': recon1,
    'reconstructed_pattern2': recon2,
    'reconstructed_pattern3': recon3,
    'fused_output': fused,
    'target_reference': target
}

for name, audio_data in audio_files.items():
    # Crop to match shortest length (360000)
    if audio_data.shape[1] > 360000:
        audio_data = audio_data[:, :360000]
    
    output_path = f'{OUTPUT_DIR}/{name}_epoch{EPOCH}.wav'
    sf.write(output_path, audio_data.T, SAMPLE_RATE)
    print(f"✓ Saved: {output_path}")

# ============================================================
# 2. WAVEFORM COMPARISON VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("GENERATING WAVEFORM VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(4, 2, figsize=(16, 12))
fig.suptitle(f'Multi-Pattern Fusion Results - Epoch {EPOCH}', fontsize=16, fontweight='bold')

# Helper to plot stereo waveform
def plot_waveform(ax, audio, title, sr=SAMPLE_RATE):
    if audio.shape[1] > 360000:
        audio = audio[:, :360000]
    times = np.arange(audio.shape[1]) / sr
    ax.plot(times, audio[0], alpha=0.7, linewidth=0.5, label='Left')
    ax.plot(times, audio[1], alpha=0.7, linewidth=0.5, label='Right')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

# Plot all patterns
plot_waveform(axes[0, 0], input1, 'Input Pattern 1')
plot_waveform(axes[0, 1], input2, 'Input Pattern 2')
plot_waveform(axes[1, 0], input3, 'Input Pattern 3')
plot_waveform(axes[1, 1], recon1, 'Reconstructed Pattern 1')
plot_waveform(axes[2, 0], recon2, 'Reconstructed Pattern 2')
plot_waveform(axes[2, 1], recon3, 'Reconstructed Pattern 3')
plot_waveform(axes[3, 0], fused, 'Fused Output (Transformer)')
plot_waveform(axes[3, 1], target[:, :360000], 'Target Reference')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/waveforms_epoch{EPOCH}.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/waveforms_epoch{EPOCH}.png")
plt.close()

# ============================================================
# 3. SPECTROGRAM COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("GENERATING SPECTROGRAM VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f'Spectrogram Comparison - Epoch {EPOCH}', fontsize=16, fontweight='bold')

def plot_spectrogram(ax, audio, title, sr=SAMPLE_RATE):
    if audio.shape[1] > 360000:
        audio = audio[:, :360000]
    # Use mono (average channels) for spectrogram
    mono = np.mean(audio, axis=0)
    D = librosa.stft(mono)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='viridis')
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(0, 8000)  # Focus on lower frequencies
    plt.colorbar(img, ax=ax, format='%+2.0f dB')

plot_spectrogram(axes[0, 0], input1, 'Input Pattern 1')
plot_spectrogram(axes[0, 1], input2, 'Input Pattern 2')
plot_spectrogram(axes[1, 0], fused, 'Fused Output (Model)')
plot_spectrogram(axes[1, 1], target[:, :360000], 'Target Reference')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/spectrograms_epoch{EPOCH}.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/spectrograms_epoch{EPOCH}.png")
plt.close()

# ============================================================
# 4. RECONSTRUCTION ERROR ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("COMPUTING RECONSTRUCTION ERRORS")
print("=" * 60)

# Crop inputs to match reconstructed length
input1_crop = input1[:, :360000]
input2_crop = input2[:, :360000]
input3_crop = input3[:, :360000]
target_crop = target[:, :360000]

# MSE for each pattern reconstruction
mse1 = np.mean((input1_crop - recon1) ** 2)
mse2 = np.mean((input2_crop - recon2) ** 2)
mse3 = np.mean((input3_crop - recon3) ** 2)
mse_fusion = np.mean((target_crop - fused) ** 2)

print(f"Pattern 1 reconstruction MSE: {mse1:.6f}")
print(f"Pattern 2 reconstruction MSE: {mse2:.6f}")
print(f"Pattern 3 reconstruction MSE: {mse3:.6f}")
print(f"Fusion prediction MSE: {mse_fusion:.6f}")

# Plot error distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Reconstruction Error Analysis - Epoch {EPOCH}', fontsize=16, fontweight='bold')

def plot_error(ax, original, reconstructed, title):
    error = original - reconstructed
    times = np.arange(error.shape[1]) / SAMPLE_RATE
    ax.plot(times, error[0], alpha=0.7, linewidth=0.5, label='Left channel')
    ax.plot(times, error[1], alpha=0.7, linewidth=0.5, label='Right channel')
    ax.set_title(f'{title}\nMSE: {np.mean(error**2):.6f}', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

plot_error(axes[0, 0], input1_crop, recon1, 'Pattern 1 Reconstruction Error')
plot_error(axes[0, 1], input2_crop, recon2, 'Pattern 2 Reconstruction Error')
plot_error(axes[1, 0], input3_crop, recon3, 'Pattern 3 Reconstruction Error')
plot_error(axes[1, 1], target_crop, fused, 'Fusion Prediction Error')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/reconstruction_errors_epoch{EPOCH}.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/reconstruction_errors_epoch{EPOCH}.png")
plt.close()

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n✓ Generated 8 audio files in {OUTPUT_DIR}/")
print(f"✓ Generated 3 visualization images in {OUTPUT_DIR}/")
print(f"\nRecommended listening order:")
print(f"  1. input_pattern1_epoch{EPOCH}.wav")
print(f"  2. input_pattern2_epoch{EPOCH}.wav")
print(f"  3. input_pattern3_epoch{EPOCH}.wav")
print(f"  4. fused_output_epoch{EPOCH}.wav (Model's fusion)")
print(f"  5. target_reference_epoch{EPOCH}.wav (Ground truth)")
print("\n" + "=" * 60)
