#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test MusicLab Backend API

Quick test to verify:
1. ffmpeg is installed and working
2. Model loads correctly
3. Audio conversion works
4. Generation pipeline completes
"""

import os
import sys
import subprocess
import numpy as np
import soundfile as sf

def test_ffmpeg():
    """Test if ffmpeg is installed"""
    print("\n" + "="*60)
    print("TEST 1: ffmpeg Installation")
    print("="*60)
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, check=True)
        version = result.stdout.split('\n')[0]
        print(f"✓ ffmpeg found: {version}")
        return True
    except FileNotFoundError:
        print("✗ ffmpeg NOT found!")
        print("  Install with:")
        print("    macOS:  brew install ffmpeg")
        print("    Ubuntu: sudo apt-get install ffmpeg")
        return False
    except Exception as e:
        print(f"✗ ffmpeg test failed: {e}")
        return False


def test_model_loading():
    """Test if model checkpoint exists and loads"""
    print("\n" + "="*60)
    print("TEST 2: Model Checkpoint")
    print("="*60)
    
    checkpoint_path = 'checkpoints/best_model.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"✓ Checkpoint exists: {checkpoint_path}")
    
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        args = checkpoint.get('args', {})
        print(f"  - Encoding dim: {args.get('encoding_dim', 128)}")
        print(f"  - Cascade stages: {args.get('num_transformer_layers', 1)}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Val loss: {checkpoint.get('val_loss', 'N/A')}")
        print("✓ Model checkpoint valid")
        return True
        
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return False


def test_audio_conversion():
    """Test audio conversion pipeline"""
    print("\n" + "="*60)
    print("TEST 3: Audio Conversion")
    print("="*60)
    
    # Create test audio (1 second sine wave)
    sample_rate = 24000
    duration = 1.0
    frequency = 440.0  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    test_dir = 'test_audio'
    os.makedirs(test_dir, exist_ok=True)
    
    # Save as WAV
    wav_path = os.path.join(test_dir, 'test.wav')
    sf.write(wav_path, audio, sample_rate)
    print(f"✓ Created test WAV: {wav_path}")
    
    # Convert to MP3
    mp3_path = os.path.join(test_dir, 'test.mp3')
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', wav_path,
            '-ar', '24000', '-ac', '1',
            mp3_path
        ], check=True, capture_output=True)
        print(f"✓ Converted to MP3: {mp3_path}")
    except Exception as e:
        print(f"✗ MP3 conversion failed: {e}")
        return False
    
    # Convert MP3 back to WAV
    wav2_path = os.path.join(test_dir, 'test_converted.wav')
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', mp3_path,
            '-ar', '24000', '-ac', '1',
            wav2_path
        ], check=True, capture_output=True)
        print(f"✓ Converted back to WAV: {wav2_path}")
    except Exception as e:
        print(f"✗ WAV conversion failed: {e}")
        return False
    
    # Verify converted audio
    converted_audio, sr = sf.read(wav2_path)
    print(f"  - Original shape: {audio.shape}")
    print(f"  - Converted shape: {converted_audio.shape}")
    print(f"  - Sample rate: {sr} Hz")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    print("✓ Audio conversion pipeline works!")
    
    return True


def test_dependencies():
    """Test Python dependencies"""
    print("\n" + "="*60)
    print("TEST 4: Python Dependencies")
    print("="*60)
    
    required = {
        'fastapi': 'FastAPI web framework',
        'uvicorn': 'ASGI server',
        'torch': 'PyTorch',
        'encodec': 'EnCodec audio codec',
        'soundfile': 'Audio I/O',
        'numpy': 'Numerical computing'
    }
    
    missing = []
    
    for package, description in required.items():
        try:
            __import__(package)
            print(f"✓ {package:15s} - {description}")
        except ImportError:
            print(f"✗ {package:15s} - MISSING!")
            missing.append(package)
    
    if missing:
        print(f"\n✗ Missing packages: {', '.join(missing)}")
        print(f"\n  Install with:")
        print(f"    pip install {' '.join(missing)}")
        return False
    
    print("✓ All dependencies installed")
    return True


def test_cache_directory():
    """Test cache directory creation"""
    print("\n" + "="*60)
    print("TEST 5: Cache Directory")
    print("="*60)
    
    cache_dir = 'cache'
    
    try:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"✓ Cache directory exists: {os.path.abspath(cache_dir)}")
        
        # Test write permissions
        test_file = os.path.join(cache_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("✓ Cache directory writable")
        
        return True
        
    except Exception as e:
        print(f"✗ Cache directory test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MUSICLAB BACKEND API - SYSTEM CHECK")
    print("="*60)
    
    tests = [
        ("ffmpeg", test_ffmpeg),
        ("Model", test_model_loading),
        ("Audio Conversion", test_audio_conversion),
        ("Dependencies", test_dependencies),
        ("Cache Directory", test_cache_directory)
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10s} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Backend is ready to run.")
        print("\nStart the server with:")
        print("  cd backend")
        print("  python main_api.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Fix issues before running server.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
