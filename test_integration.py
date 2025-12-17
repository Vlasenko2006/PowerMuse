#!/usr/bin/env python3
"""
Quick Integration Test - MusicLab Complete System
Tests the full pipeline: upload → generate → download
"""

import requests
import time
import os

API_URL = "http://localhost:8001"

print("="*60)
print("MUSICLAB INTEGRATION TEST")
print("="*60)

# Test 1: Health Check
print("\n[1/4] Testing health endpoint...")
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Server healthy:")
        print(f"  - Model loaded: {data['model_loaded']}")
        print(f"  - EnCodec loaded: {data['encodec_loaded']}")
        print(f"  - Device: {data['device']}")
    else:
        print(f"✗ Health check failed: {response.status_code}")
        exit(1)
except Exception as e:
    print(f"✗ Cannot connect to server: {e}")
    print("\nMake sure backend is running:")
    print("  ps aux | grep main_api")
    exit(1)

# Test 2: Check for test audio files
print("\n[2/4] Checking for test audio files...")
test_files = [
    "inference_outputs/predicted_audio_variant_0.wav",
    "inference_outputs/predicted_audio_variant_1.wav"
]

audio_files = []
for f in test_files:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"✓ Found: {f} ({size_mb:.1f} MB)")
        audio_files.append(f)
    else:
        print(f"✗ Not found: {f}")

if len(audio_files) < 2:
    print("\n⚠️  Need at least 2 audio files for testing")
    print("Creating test audio files...")
    
    import numpy as np
    import soundfile as sf
    
    os.makedirs("test_audio", exist_ok=True)
    
    # Create 20-second test audio files
    sample_rate = 24000
    duration = 20.0
    
    for i in range(2):
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Different frequencies for each track
        freq = 440.0 + (i * 110)  # A4 and C5
        audio = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
        
        filepath = f"test_audio/track{i+1}.wav"
        sf.write(filepath, audio, sample_rate)
        print(f"✓ Created: {filepath}")
        audio_files.append(filepath)

# Test 3: Upload and Generate
print("\n[3/4] Testing music generation...")
print(f"  Track 1: {audio_files[0]}")
print(f"  Track 2: {audio_files[1]}")

try:
    with open(audio_files[0], 'rb') as f1, open(audio_files[1], 'rb') as f2:
        files = {
            'track1': ('track1.wav', f1, 'audio/wav'),
            'track2': ('track2.wav', f2, 'audio/wav')
        }
        data = {
            'start_time_1': 0.0,
            'end_time_1': 16.0,
            'start_time_2': 0.0,
            'end_time_2': 16.0
        }
        
        print("  Uploading files...")
        response = requests.post(f"{API_URL}/api/generate", files=files, data=data, timeout=30)
        
        if response.status_code != 200:
            print(f"✗ Generation failed: {response.status_code}")
            print(response.text)
            exit(1)
        
        result = response.json()
        job_id = result['job_id']
        print(f"✓ Job created: {job_id}")
        
        # Poll for status
        print("  Waiting for generation to complete...")
        max_wait = 120  # 2 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            response = requests.get(f"{API_URL}/api/status/{job_id}")
            status = response.json()
            
            progress = status['progress']
            message = status['message']
            print(f"  Progress: {progress}% - {message}", end='\r')
            
            if status['status'] == 'completed':
                print(f"\n✓ Generation completed!")
                break
            elif status['status'] == 'failed':
                print(f"\n✗ Generation failed: {message}")
                exit(1)
            
            time.sleep(1)
        else:
            print(f"\n✗ Timeout after {max_wait}s")
            exit(1)

except Exception as e:
    print(f"\n✗ Error during generation: {e}")
    exit(1)

# Test 4: Download result
print("\n[4/4] Testing download...")
try:
    response = requests.get(f"{API_URL}/api/download/{job_id}", timeout=30)
    
    if response.status_code != 200:
        print(f"✗ Download failed: {response.status_code}")
        exit(1)
    
    output_file = f"test_generated_{job_id}.wav"
    with open(output_file, 'wb') as f:
        f.write(response.content)
    
    size_mb = len(response.content) / (1024 * 1024)
    print(f"✓ Downloaded: {output_file} ({size_mb:.2f} MB)")
    
    # Verify it's a valid WAV file
    import soundfile as sf
    audio, sr = sf.read(output_file)
    duration = len(audio) / sr
    print(f"  - Sample rate: {sr} Hz")
    print(f"  - Duration: {duration:.1f}s")
    print(f"  - Samples: {len(audio)}")
    
except Exception as e:
    print(f"✗ Download error: {e}")
    exit(1)

print("\n" + "="*60)
print("🎉 ALL TESTS PASSED!")
print("="*60)
print(f"\nGenerated audio saved to: {output_file}")
print("You can play it with:")
print(f"  afplay {output_file}  # macOS")
print(f"  play {output_file}    # Linux with sox")
print("\nBackend is fully operational and ready for production use!")
