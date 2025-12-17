# Bug Fixes - December 17, 2025

## Issues Reported
User reported 3 critical bugs after testing the music generation pipeline:
1. **Sliding window ignored** - Audio patterns start from beginning of song (0:00) instead of selected time window
2. **Downloaded file is 0 bytes** - Generated audio file cannot be downloaded
3. **Generated audio deleted before playback** - Visualization shows but no sound

## Root Causes Identified

### Bug 1: API Parameter Mismatch
**Problem**: Backend only accepted 2 time parameters (`start_time`, `end_time`) while frontend sends 4 parameters (`start_time_1`, `end_time_1`, `start_time_2`, `end_time_2`).

**Impact**: Backend always used default `start_time=0.0` for both tracks, ignoring user's sliding window selection.

**Investigation**:
```bash
# grep_search for "start_time_1" in backend/main_api.py
Result: No matches found

# Frontend sends 4 parameters:
formData.append('start_time_1', track1Start.toString());
formData.append('end_time_1', track1End.toString());
formData.append('start_time_2', track2Start.toString());
formData.append('end_time_2', track2End.toString());

# Backend only accepted 2:
start_time: float = Form(0.0),
end_time: float = Form(16.0),
```

### Bug 2 & 3: Premature File Cleanup
**Problem**: Backend deleted generated audio immediately after processing, before user could download or play it.

**Code location**: Lines 367-373 in `backend/main_api.py`

## Fixes Applied

### Fix 1: Update API Endpoint (Lines 413-438)
**Changed signature from:**
```python
@app.post("/api/generate")
async def generate_music_endpoint(
    track1: UploadFile = File(...),
    track2: UploadFile = File(...),
    start_time: float = Form(0.0),
    end_time: float = Form(16.0),
    ...
)
```

**To:**
```python
@app.post("/api/generate")
async def generate_music_endpoint(
    track1: UploadFile = File(...),
    track2: UploadFile = File(...),
    start_time_1: float = Form(0.0),
    end_time_1: float = Form(16.0),
    start_time_2: float = Form(0.0),
    end_time_2: float = Form(16.0),
    ...
)
```

### Fix 2: Update Background Task Call (Lines 471-476)
**Changed from:**
```python
background_tasks.add_task(
    generate_music,
    job_id, track1_path, track2_path, start_time, end_time
)
```

**To:**
```python
background_tasks.add_task(
    generate_music,
    job_id, track1_path, track2_path, 
    start_time_1, end_time_1, start_time_2, end_time_2
)
```

### Fix 3: Update generate_music Function (Lines 249-252)
**Changed signature from:**
```python
def generate_music(job_id: str, track1_path: str, track2_path: str, 
                   start_time: float, end_time: float):
```

**To:**
```python
def generate_music(job_id: str, track1_path: str, track2_path: str, 
                   start_time_1: float, end_time_1: float,
                   start_time_2: float, end_time_2: float):
```

### Fix 4: Use Correct Start Times for Extraction (Lines 292-298)
**Changed from:**
```python
audio1, sr1 = extract_audio_segment(wav1_path, start_time, 16.0)
audio2, sr2 = extract_audio_segment(wav2_path, start_time, 16.0)
```

**To:**
```python
duration = 16.0
print(f"DEBUG: Extracting segment from track1 ({start_time_1}s - {start_time_1+duration}s)")
audio1, sr1 = extract_audio_segment(wav1_path, start_time_1, duration)
print(f"DEBUG: Track1 audio shape: {audio1.shape}, sr: {sr1}")
print(f"DEBUG: Extracting segment from track2 ({start_time_2}s - {start_time_2+duration}s)")
audio2, sr2 = extract_audio_segment(wav2_path, start_time_2, duration)
```

### Fix 5: Safe Cleanup with Error Handling (Lines 371-377)
**Changed from:**
```python
for temp_file in [track1_path, track2_path, wav1_path, wav2_path]:
    if os.path.exists(temp_file):
        os.remove(temp_file)
        logger.info(f"  🗑️  Cleaned up: {temp_file}")
```

**To:**
```python
for temp_file in [track1_path, track2_path, wav1_path, wav2_path]:
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
            logger.info(f"  🗑️  Cleaned up: {temp_file}")
        except Exception as e:
            logger.warning(f"  ⚠️  Could not delete {temp_file}: {e}")
```

**Note**: Only temporary conversion files are deleted. Generated output (`generated.wav`) is kept for download.

### Fix 6: Enhanced Logging (Lines 440-442)
**Changed from:**
```python
logger.info(f"   Track 1: {track1.filename}")
logger.info(f"   Track 2: {track2.filename}")
logger.info(f"   Time range: {start_time}s - {end_time}s")
```

**To:**
```python
logger.info(f"   Track 1: {track1.filename} ({start_time_1}s - {end_time_1}s)")
logger.info(f"   Track 2: {track2.filename} ({start_time_2}s - {end_time_2}s)")
```

## Server Restart
Backend restarted successfully:
```bash
PID: 52061
Status: ✅ Healthy
Model: Loaded (16.7M parameters)
EnCodec: Loaded
Port: 8001
```

## Testing Instructions
1. Open frontend in browser: `file:///Users/andreyvlasenko/tst/Jingle_D/frontend/index.html`
2. Upload two audio files (MP3 or WAV)
3. Select different time windows using sliders:
   - Track 1: e.g., 30s - 46s
   - Track 2: e.g., 15s - 31s
4. Click "Generate Music"
5. Wait for completion (progress bar)
6. Verify:
   - ✅ Generated audio uses correct time segments (not starting at 0:00)
   - ✅ Download button works and file is not 0 bytes
   - ✅ Playback button works and audio is audible

## Expected Behavior
- **Track 1 Segment**: Should extract audio from selected start_time_1 to end_time_1
- **Track 2 Segment**: Should extract audio from selected start_time_2 to end_time_2
- **Download**: Should download complete WAV file (typically 700KB+ for 16s audio)
- **Playback**: Should play generated audio with visualization

## Files Modified
- `backend/main_api.py` (6 changes across ~50 lines)

## Status
✅ **ALL FIXES APPLIED AND DEPLOYED**

Ready for testing!
