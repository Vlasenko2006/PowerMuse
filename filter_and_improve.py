#!/usr/bin/env python3
"""
Apply spectral filtering to remove parasitic frequency outliers from fused output.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf

# Configuration
SAMPLE_RATE = 22050
EPOCH = 50
INPUT_DIR = 'mus_output'
OUTPUT_DIR = 'audio_output'

print("=" * 60)
print("SPECTRAL FILTERING - REMOVING PARASITIC FREQUENCIES")
print("=" * 60)

# Load fused output and target
print("\nLoading data...")
fused = np.load(f'{INPUT_DIR}/output_fused_epoch_{EPOCH}.npy')
target = np.load(f'{INPUT_DIR}/target_reference_epoch_{EPOCH}.npy')[:, :360000]

print(f"✓ Loaded fused output: {fused.shape}")
print(f"✓ Loaded target reference: {target.shape}")

# ============================================================
# 1. ANALYZE SPECTRAL CONTENT
# ============================================================
print("\n" + "=" * 60)
print("ANALYZING SPECTRAL CONTENT")
print("=" * 60)

def analyze_spectrum(audio, label, sample_rate=SAMPLE_RATE):
    """Compute and analyze power spectral density."""
    freqs, psd = signal.welch(audio[0], fs=sample_rate, nperseg=4096)
    peaks, props = signal.find_peaks(10*np.log10(psd), height=-50, prominence=5)
    
    print(f"\n{label}:")
    print(f"  Total spectral peaks: {len(peaks)}")
    if len(peaks) > 0:
        print(f"  Peak frequencies (Hz):")
        for i, peak in enumerate(peaks[:10]):
            print(f"    {freqs[peak]:6.1f} Hz: {10*np.log10(psd[peak]):5.1f} dB")
        if len(peaks) > 10:
            print(f"    ... ({len(peaks) - 10} more peaks)")
    
    return freqs, psd, peaks

freqs_fused, psd_fused, peaks_fused = analyze_spectrum(fused, "Fused Output (Original)")
freqs_target, psd_target, peaks_target = analyze_spectrum(target, "Target Reference")

# ============================================================
# 2. DESIGN AND APPLY FILTERS
# ============================================================
print("\n" + "=" * 60)
print("DESIGNING FILTERS")
print("=" * 60)

# Strategy 1: Notch filters for specific parasitic frequencies
# Identify frequencies present in fused but not in target
parasitic_freqs = []
for peak in peaks_fused:
    freq = freqs_fused[peak]
    # Check if this frequency is NOT present in target
    is_parasitic = True
    for target_peak in peaks_target:
        if abs(freqs_target[target_peak] - freq) < 100:  # Within 100 Hz tolerance
            is_parasitic = False
            break
    if is_parasitic:
        parasitic_freqs.append(freq)

print(f"\nIdentified {len(parasitic_freqs)} parasitic frequencies:")
for freq in parasitic_freqs[:10]:
    print(f"  {freq:.1f} Hz")

# Strategy 2: Bandpass filter (keep only music-relevant frequencies)
# Most music content is between 20 Hz - 8000 Hz
print("\nApplying filters...")

filtered_fused = fused.copy()

# Apply notch filters for parasitic frequencies
for freq in parasitic_freqs:
    if 50 < freq < 10000:  # Only filter reasonable frequencies
        # Design notch filter
        Q = 30.0  # Quality factor
        b, a = signal.iirnotch(freq, Q, SAMPLE_RATE)
        
        # Apply to both channels
        filtered_fused[0] = signal.filtfilt(b, a, filtered_fused[0])
        filtered_fused[1] = signal.filtfilt(b, a, filtered_fused[1])
        print(f"  Applied notch filter at {freq:.1f} Hz (Q={Q})")

# Apply gentle lowpass filter to remove high-frequency artifacts
# Butterworth filter with cutoff at 8 kHz
print("\nApplying lowpass filter...")
sos = signal.butter(4, 8000, 'lowpass', fs=SAMPLE_RATE, output='sos')
filtered_fused[0] = signal.sosfiltfilt(sos, filtered_fused[0])
filtered_fused[1] = signal.sosfiltfilt(sos, filtered_fused[1])
print(f"  Applied 4th-order Butterworth lowpass at 8000 Hz")

# Apply gentle highpass filter to remove DC offset and very low rumble
print("\nApplying highpass filter...")
sos_hp = signal.butter(2, 20, 'highpass', fs=SAMPLE_RATE, output='sos')
filtered_fused[0] = signal.sosfiltfilt(sos_hp, filtered_fused[0])
filtered_fused[1] = signal.sosfiltfilt(sos_hp, filtered_fused[1])
print(f"  Applied 2nd-order Butterworth highpass at 20 Hz")

# ============================================================
# 3. ANALYZE FILTERED RESULT
# ============================================================
print("\n" + "=" * 60)
print("ANALYZING FILTERED OUTPUT")
print("=" * 60)

freqs_filtered, psd_filtered, peaks_filtered = analyze_spectrum(filtered_fused, "Fused Output (Filtered)")

# Compute MSE improvement
mse_original = np.mean((target - fused) ** 2)
mse_filtered = np.mean((target - filtered_fused) ** 2)

print(f"\nMSE Comparison:")
print(f"  Original fused output: {mse_original:.6f}")
print(f"  Filtered fused output: {mse_filtered:.6f}")
print(f"  Improvement: {((mse_original - mse_filtered) / mse_original * 100):.2f}%")

# ============================================================
# 4. SAVE FILTERED AUDIO
# ============================================================
print("\n" + "=" * 60)
print("SAVING FILTERED AUDIO")
print("=" * 60)

output_path = f'{OUTPUT_DIR}/fused_output_filtered_epoch{EPOCH}.wav'
sf.write(output_path, filtered_fused.T, SAMPLE_RATE)
print(f"✓ Saved: {output_path}")

# ============================================================
# 5. GENERATE COMPARISON VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Spectral comparison
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('Spectral Filtering Results', fontsize=16, fontweight='bold')

# Plot 1: Power Spectral Density
axes[0].semilogy(freqs_fused, psd_fused, label='Original Fused', alpha=0.7, linewidth=1.5)
axes[0].semilogy(freqs_filtered, psd_filtered, label='Filtered Fused', alpha=0.7, linewidth=1.5)
axes[0].semilogy(freqs_target, psd_target, label='Target Reference', alpha=0.7, linewidth=1.5)
axes[0].plot(freqs_fused[peaks_fused], psd_fused[peaks_fused], 'rx', markersize=8, label='Parasitic Peaks')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('PSD')
axes[0].set_title('Power Spectral Density Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 10000)

# Plot 2: dB scale
axes[1].plot(freqs_fused, 10*np.log10(psd_fused), label='Original Fused', alpha=0.7, linewidth=1.5)
axes[1].plot(freqs_filtered, 10*np.log10(psd_filtered), label='Filtered Fused', alpha=0.7, linewidth=1.5)
axes[1].plot(freqs_target, 10*np.log10(psd_target), label='Target Reference', alpha=0.7, linewidth=1.5)
axes[1].plot(freqs_fused[peaks_fused], 10*np.log10(psd_fused[peaks_fused]), 'rx', markersize=8, label='Removed Peaks')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Power (dB)')
axes[1].set_title('Power Spectrum (dB Scale) - Before and After Filtering')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 10000)
axes[1].set_ylim(-80, 0)

# Plot 3: Waveform comparison
axes[2].plot(np.arange(len(fused[0])) / SAMPLE_RATE, fused[0], alpha=0.5, linewidth=0.5, label='Original')
axes[2].plot(np.arange(len(filtered_fused[0])) / SAMPLE_RATE, filtered_fused[0], alpha=0.5, linewidth=0.5, label='Filtered')
axes[2].plot(np.arange(len(target[0])) / SAMPLE_RATE, target[0], alpha=0.5, linewidth=0.5, label='Target')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Amplitude')
axes[2].set_title('Waveform Comparison (Left Channel)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(0, 5)  # Show first 5 seconds

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/filtering_comparison_epoch{EPOCH}.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/filtering_comparison_epoch{EPOCH}.png")
plt.close()

# Spectrogram comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Spectrogram Comparison - Filtering Effect', fontsize=16, fontweight='bold')

def plot_spectrogram(ax, audio, title):
    import librosa
    import librosa.display
    mono = np.mean(audio, axis=0)
    D = librosa.stft(mono)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, sr=SAMPLE_RATE, x_axis='time', y_axis='hz', ax=ax, cmap='viridis')
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(0, 8000)
    plt.colorbar(img, ax=ax, format='%+2.0f dB')

plot_spectrogram(axes[0], fused, 'Original Fused Output')
plot_spectrogram(axes[1], filtered_fused, 'Filtered Fused Output')
plot_spectrogram(axes[2], target, 'Target Reference')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/spectrograms_filtered_epoch{EPOCH}.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/spectrograms_filtered_epoch{EPOCH}.png")
plt.close()

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n✓ Removed {len(parasitic_freqs)} parasitic frequency peaks")
print(f"✓ Applied lowpass filter at 8000 Hz")
print(f"✓ Applied highpass filter at 20 Hz")
print(f"\nSpectral peaks reduced:")
print(f"  Before: {len(peaks_fused)} peaks")
print(f"  After: {len(peaks_filtered)} peaks")
print(f"\nMSE change: {mse_original:.6f} → {mse_filtered:.6f}")
print(f"\nFiltered audio saved to:")
print(f"  {output_path}")
print("\nRecommended listening comparison:")
print(f"  1. {OUTPUT_DIR}/fused_output_epoch{EPOCH}.wav (original)")
print(f"  2. {OUTPUT_DIR}/fused_output_filtered_epoch{EPOCH}.wav (filtered)")
print(f"  3. {OUTPUT_DIR}/target_reference_epoch{EPOCH}.wav (target)")
print("\n" + "=" * 60)
