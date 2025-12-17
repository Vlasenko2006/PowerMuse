# MusicLab Backend API - Implementation Summary

**Date:** December 17, 2025  
**Purpose:** Complete backend API for MusicLab music generation service

---

## Q1: Model Input/Output Format Analysis

### From `inference_cascade.py` Analysis:

**Input Format:**
- **File format:** WAV files (any format converted to WAV)
- **Sample rate:** 24,000 Hz (24 kHz)
- **Channels:** Mono (stereo automatically converted to mono)
- **Duration:** 16 seconds per segment
- **Data type:** float32, normalized to [-1.0, 1.0]
- **Tensor shape:** `[1, 1, samples]` where samples = 24000 * 16 = 384,000

**Model Pipeline:**
```
Audio File (.mp3/.m4a/.wav)
    ↓
WAV 24kHz Mono (via ffmpeg)
    ↓
16-second Segment Extraction
    ↓
Numpy Array [384000 samples]
    ↓
Torch Tensor [1, 1, 384000]
    ↓
EnCodec Encoder → [1, 128, T]
    ↓
SimpleTransformer Model → [1, 128, T]
    ↓
EnCodec Decoder → [1, 1, 384000]
    ↓
Output WAV File
```

**EnCodec Configuration:**
- Model: `encodec_model_24khz()`
- Bandwidth: 6.0
- Encoding dimension: 128 channels
- Frozen parameters (encoder + decoder)

**Model Configuration:**
- Architecture: SimpleTransformer (cascade or single stage)
- Encoding dim: 128
- Attention heads: 8
- Transformer layers: 4 (internal)
- Cascade stages: 1-2+ (num_transformer_layers)
- Checkpoint: `checkpoints/best_model.pt`

**Output Format:**
- **File format:** WAV (PCM_16)
- **Sample rate:** 24,000 Hz
- **Channels:** Mono
- **Duration:** 16 seconds
- **Normalization:** Clipped to [-1.0, 1.0]

---

## Q2: Audio Conversion Method for AWS Deployment

### Selected Method: **ffmpeg** (Best for AWS)

**Why ffmpeg?**
✅ **Universal compatibility:** Handles ALL formats (mp3, m4a, flac, ogg, wav, etc.)  
✅ **AWS-friendly:** Available via package managers (yum/apt)  
✅ **Fast & reliable:** C-based, optimized performance  
✅ **Docker-ready:** Easy to include in containers  
✅ **No Python dependencies:** Standalone binary  
✅ **Industry standard:** Used by YouTube, Netflix, AWS MediaConvert

**Implementation:**
```python
def convert_to_wav(input_path: str, output_path: str, sample_rate: int = 24000):
    """Convert any audio format to WAV 24kHz mono using ffmpeg"""
    subprocess.run([
        'ffmpeg',
        '-i', input_path,        # Input file
        '-ar', str(sample_rate), # Audio sample rate
        '-ac', '1',              # Mono channel
        '-y',                    # Overwrite output
        output_path
    ], check=True, capture_output=True)
```

**AWS Installation:**
```bash
# Amazon Linux / EC2
sudo yum install ffmpeg -y

# Ubuntu / Debian
sudo apt-get install ffmpeg -y

# Docker
RUN apt-get update && apt-get install -y ffmpeg
```

**Alternative Considered (Rejected):**
- ❌ `pydub`: Requires ffmpeg anyway + extra Python overhead
- ❌ `torchaudio`: Limited format support, heavy dependency
- ❌ `librosa`: Slower, inconsistent sample rate conversion

---

## Q3: File Caching & Storage Strategy

### Cache Directory Structure:
```
cache/
├── music_{job_id_1}/
│   ├── track1.mp3         # Original upload (deleted after WAV conversion)
│   ├── track2.m4a         # Original upload (deleted after WAV conversion)
│   ├── track1.wav         # Converted WAV (deleted after encoding)
│   ├── track2.wav         # Converted WAV (deleted after encoding)
│   └── generated.wav      # Final output (kept until download + cleanup)
├── music_{job_id_2}/
│   └── ...
```

### Storage Lifecycle:

**1. Upload Phase:**
```python
cache_dir = f"cache/music_{job_id}"
os.makedirs(cache_dir, exist_ok=True)

# Save uploads with original extension
track1_path = os.path.join(cache_dir, f"track1{original_ext}")
track2_path = os.path.join(cache_dir, f"track2{original_ext}")
```

**2. Processing Phase:**
```python
# Convert to WAV
wav1_path = os.path.join(cache_dir, 'track1.wav')
wav2_path = os.path.join(cache_dir, 'track2.wav')
convert_to_wav(track1_path, wav1_path)
convert_to_wav(track2_path, wav2_path)

# After conversion: DELETE originals
os.remove(track1_path)
os.remove(track2_path)
```

**3. Generation Phase:**
```python
# Extract segments, encode, run model, decode
output_path = os.path.join(cache_dir, 'generated.wav')
sf.write(output_path, predicted_audio, 24000)

# After generation: DELETE temporary WAVs
os.remove(wav1_path)
os.remove(wav2_path)

# KEEP only generated.wav
jobs_db[job_id]['output_path'] = output_path
```

**4. Cleanup Phase:**
```python
# After user downloads or 24h expiration
DELETE /api/cleanup/{job_id}
→ shutil.rmtree(f"cache/music_{job_id}")
→ del jobs_db[job_id]
```

### Automatic Cleanup (Production):
```python
# Add to backend/main_api.py
import asyncio
from datetime import datetime, timedelta

async def cleanup_old_jobs():
    """Background task to cleanup expired jobs"""
    while True:
        await asyncio.sleep(3600)  # Run every hour
        
        now = datetime.now()
        expired = []
        
        for job_id, job in jobs_db.items():
            created = datetime.fromisoformat(job['created_at'])
            if now - created > timedelta(hours=24):
                expired.append(job_id)
        
        for job_id in expired:
            cache_dir = f"cache/music_{job_id}"
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            del jobs_db[job_id]
            logger.info(f"🗑️  Auto-cleaned expired job: {job_id}")

# Start at app startup
@app.on_event("startup")
async def startup_cleanup():
    asyncio.create_task(cleanup_old_jobs())
```

---

## Q4: API Routes & File Serving

### Route Mapping:

**✓ MATCHES REQUIREMENT:**
```python
@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    """Download generated audio file"""
    # Verify job exists and is complete
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Not ready")
    
    # Get output path from database
    output_path = jobs_db[job_id]['output_path']
    
    # Serve file
    return FileResponse(
        output_path,
        media_type='audio/wav',
        filename='generated_music.wav'
    )
```

**Storage Match:**
```python
# Generation pipeline stores path correctly:
jobs_db[job_id]['output_path'] = f"cache/music_{job_id}/generated.wav"

# Download endpoint retrieves it correctly:
output_path = jobs_db[job_id]['output_path']
return FileResponse(output_path, ...)
```

**Complete API Endpoints:**
- `POST /api/generate` → Upload tracks, start generation
- `GET /api/status/{job_id}` → Poll generation progress
- `GET /api/download/{job_id}` → Download completed audio ✓
- `DELETE /api/cleanup/{job_id}` → Manual cleanup
- `GET /health` → Health check for AWS

---

## Q5: Chatbot Integration

### Current Status: **INACTIVE** ✓

**Prepared for Future Activation:**
```python
# backend/main_api.py (commented out for now)
"""
from chatbot_analyzer import MusicLabChatbot

chatbots: Dict[str, MusicLabChatbot] = {}

@app.post("/api/chat")
async def chat_endpoint(job_id: str, question: str):
    # RAG over generation results
    # Answer questions about:
    # - How was the music generated?
    # - What features were extracted from track1/track2?
    # - What parameters were used?
    # - How to adjust the output?
    pass
"""
```

**Future Implementation Ideas:**
1. **Technical explanations:** How EnCodec encoding works
2. **Parameter suggestions:** "Try increasing start_time to 5.0"
3. **Musical analysis:** "Track 1 has strong rhythm, Track 2 has harmonic content"
4. **Troubleshooting:** "Output too quiet? Model may need retraining"

**When to Activate:**
- After Stage 2 API integration is stable
- If users request help understanding outputs
- For premium features or advanced users

---

## Complete API Workflow

### 1. Frontend Upload (index.html)
```javascript
const formData = new FormData();
formData.append('track1', track1File);
formData.append('track2', track2File);
formData.append('start_time', 0.0);
formData.append('end_time', 16.0);

const response = await fetch('http://localhost:8001/api/generate', {
    method: 'POST',
    body: formData
});

const { job_id } = await response.json();
```

### 2. Backend Processing (main_api.py)
```python
POST /api/generate
    ↓
1. Save uploads to cache/music_{job_id}/
    ↓
2. Start background task
    ↓
3. Convert to WAV (ffmpeg) → DELETE originals
    ↓
4. Extract 16s segments
    ↓
5. Encode with EnCodec
    ↓
6. Run through model
    ↓
7. Decode output
    ↓
8. Save generated.wav → DELETE temp WAVs
    ↓
9. Update jobs_db[job_id]['output_path']
```

### 3. Frontend Polling
```javascript
const pollStatus = setInterval(async () => {
    const res = await fetch(`http://localhost:8001/api/status/${job_id}`);
    const status = await res.json();
    
    updateProgressBar(status.progress); // 0-100
    updateStatusText(status.message);
    
    if (status.status === 'completed') {
        clearInterval(pollStatus);
        enableDownloadButton(job_id);
    }
}, 2000); // Poll every 2 seconds
```

### 4. Frontend Download
```javascript
function downloadMusic(job_id) {
    window.open(`http://localhost:8001/api/download/${job_id}`, '_blank');
}
```

### 5. Optional Cleanup
```javascript
// After download completes
await fetch(`http://localhost:8001/api/cleanup/${job_id}`, {
    method: 'DELETE'
});
```

---

## Deployment Checklist

### Local Development:
```bash
# 1. Install ffmpeg
brew install ffmpeg  # macOS
# or
sudo apt-get install ffmpeg  # Ubuntu

# 2. Install Python dependencies
pip install fastapi uvicorn python-multipart soundfile numpy torch encodec

# 3. Run server
cd backend
python main_api.py
# Server runs on http://localhost:8001

# 4. Test health
curl http://localhost:8001/health
```

### AWS Deployment:
```bash
# 1. EC2 Instance Setup
sudo yum install ffmpeg -y
sudo yum install python3 -y
pip3 install fastapi uvicorn python-multipart soundfile numpy torch encodec

# 2. GPU Support (optional, for faster generation)
# Use g4dn.xlarge instance with CUDA

# 3. Docker (recommended)
# See Dockerfile below
```

### Dockerfile:
```dockerfile
FROM python:3.10-slim

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Copy code
WORKDIR /app
COPY backend/ /app/backend/
COPY checkpoints/ /app/checkpoints/
COPY model_simple_transformer.py /app/

# Install dependencies
RUN pip install fastapi uvicorn python-multipart soundfile numpy torch encodec

# Expose port
EXPOSE 8001

# Run server
CMD ["python", "backend/main_api.py"]
```

### Docker Compose:
```yaml
version: '3.8'
services:
  backend:
    build: .
    ports:
      - "8001:8001"
    volumes:
      - ./cache:/app/cache
      - ./checkpoints:/app/checkpoints
    environment:
      - CUDA_VISIBLE_DEVICES=0  # If GPU available
    restart: unless-stopped
  
  frontend:
    image: nginx:alpine
    ports:
      - "3000:80"
    volumes:
      - ./frontend:/usr/share/nginx/html
    restart: unless-stopped
```

---

## Testing Commands

### 1. Test Conversion:
```bash
# Create test audio
ffmpeg -f lavfi -i "sine=frequency=440:duration=5" test.mp3

# Test conversion endpoint
curl -X POST http://localhost:8001/api/generate \
  -F "track1=@test.mp3" \
  -F "track2=@test.mp3" \
  -F "start_time=0.0" \
  -F "end_time=16.0"
```

### 2. Test Status:
```bash
# Replace {job_id} with actual job ID from generate response
curl http://localhost:8001/api/status/{job_id}
```

### 3. Test Download:
```bash
wget http://localhost:8001/api/download/{job_id} -O output.wav
```

---

## Performance Metrics

### Expected Processing Times:
- **Upload:** < 1 second (depends on file size)
- **Conversion (ffmpeg):** 1-2 seconds per file
- **Encoding (EnCodec):** 2-3 seconds
- **Model inference:** 3-5 seconds (GPU) / 30-60 seconds (CPU)
- **Decoding:** 1-2 seconds
- **Total:** ~10-15 seconds (GPU) / ~60-90 seconds (CPU)

### Storage Requirements:
- **Per job:** ~5-10 MB during processing, ~2-3 MB final output
- **Daily cleanup:** Recommended every 24 hours
- **AWS S3 option:** For long-term storage of generated music

---

## Security Considerations

### Production Recommendations:
1. **File size limits:** Max 50 MB per upload
2. **Rate limiting:** 10 requests per minute per IP
3. **File type validation:** Whitelist audio formats only
4. **Cleanup automation:** Delete files after 24 hours
5. **CORS:** Restrict to frontend domain only
6. **API keys:** Add authentication for production

### File Validation:
```python
ALLOWED_EXTENSIONS = {'.mp3', '.m4a', '.wav', '.flac', '.ogg'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

@app.post("/api/generate")
async def generate_music_endpoint(track1: UploadFile, ...):
    # Validate file extensions
    ext1 = os.path.splitext(track1.filename)[1].lower()
    if ext1 not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Invalid file format")
    
    # Validate file size
    track1.file.seek(0, 2)  # Seek to end
    size = track1.file.tell()
    track1.file.seek(0)  # Reset
    
    if size > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")
```

---

## Summary of Answers

### ✅ Q1: Model Input/Output Format
- **Input:** WAV 24kHz mono, 16-second segments, normalized float32
- **Output:** WAV 24kHz mono, 16-second generated music, PCM_16
- **Conversion:** Use **ffmpeg** (best for AWS deployment)

### ✅ Q2: File Caching
- **Location:** `cache/music_{job_id}/`
- **Lifecycle:** Upload → Convert → Process → Output → Cleanup
- **Deletion:** Automatic after processing (keep only final output)

### ✅ Q3: File Serving
- **Route:** `/api/download/{job_id}` ✓ MATCHES REQUIREMENT
- **Storage:** `jobs_db[job_id]['output_path']` ✓ MATCHES REQUIREMENT
- **Method:** FastAPI `FileResponse` with `media_type='audio/wav'`

### ✅ Q4: Chatbot
- **Status:** INACTIVE ✓ AS REQUESTED
- **Future:** RAG-based music generation assistant
- **Activation:** When Stage 2 API is stable

---

## Next Steps

1. **Test backend locally:**
   ```bash
   cd backend
   python main_api.py
   ```

2. **Update frontend to use API:**
   - Add "Submit" button
   - Implement file upload
   - Add progress polling
   - Enable download

3. **Deploy to AWS:**
   - Use Docker container
   - Configure NGINX reverse proxy
   - Setup automatic cleanup cron job

4. **Monitor & optimize:**
   - Add logging
   - Track generation times
   - Monitor disk usage
   - Profile model inference

---

**Implementation Complete!** 🎵
Backend API ready for integration with MusicLab frontend.
