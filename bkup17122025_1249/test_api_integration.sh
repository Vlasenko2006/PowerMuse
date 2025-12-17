#!/bin/bash
# Test MusicLab API Integration
# This script tests the complete frontend-backend workflow via CLI

API_URL="http://localhost:8001"
TRACK1="dataset_pairs_wav/val/pair_0116_input.wav"
TRACK2="dataset_pairs_wav/val/pair_0050_input.wav"

echo "=================================================="
echo "MusicLab API Integration Test"
echo "=================================================="
echo ""

# 1. Check if server is running
echo "1. Checking if backend server is running..."
if ps aux | grep -v grep | grep main_api.py > /dev/null; then
    echo "   ✓ Backend server is running"
else
    echo "   ✗ Backend server is NOT running!"
    echo "   Start with: /Volumes/Music_Video_Foto/conda/anaconda3/envs/PCL_copy/bin/python backend/main_api.py &"
    exit 1
fi
echo ""

# 2. Health check
echo "2. Testing health endpoint..."
HEALTH=$(curl -s $API_URL/health)
if [ $? -eq 0 ]; then
    echo "   ✓ Health check passed"
    echo "   Response: $HEALTH"
else
    echo "   ✗ Health check failed"
    exit 1
fi
echo ""

# 3. Submit generation request
echo "3. Submitting generation request..."
echo "   Track 1: $TRACK1"
echo "   Track 2: $TRACK2"
echo "   Segments: 0-16 seconds"
echo ""

RESPONSE=$(curl -s -X POST $API_URL/api/generate \
  -F "track1=@$TRACK1" \
  -F "track2=@$TRACK2" \
  -F "start_time_1=0.0" \
  -F "end_time_1=16.0" \
  -F "start_time_2=0.0" \
  -F "end_time_2=16.0")

echo "   Response: $RESPONSE"

# Extract job_id
JOB_ID=$(echo $RESPONSE | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$JOB_ID" ]; then
    echo "   ✗ Failed to get job_id"
    exit 1
fi

echo "   ✓ Job submitted with ID: $JOB_ID"
echo ""

# 4. Poll for status
echo "4. Monitoring generation progress..."
PROGRESS=0
while [ $PROGRESS -lt 100 ]; do
    sleep 2
    STATUS_RESPONSE=$(curl -s $API_URL/api/status/$JOB_ID)
    
    PROGRESS=$(echo $STATUS_RESPONSE | grep -o '"progress":[0-9]*' | cut -d':' -f2)
    STATUS=$(echo $STATUS_RESPONSE | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    MESSAGE=$(echo $STATUS_RESPONSE | grep -o '"message":"[^"]*"' | cut -d'"' -f4)
    
    echo "   Progress: $PROGRESS% | Status: $STATUS | $MESSAGE"
    
    if [ "$STATUS" = "failed" ]; then
        echo "   ✗ Generation failed!"
        exit 1
    fi
    
    if [ "$STATUS" = "completed" ]; then
        break
    fi
done

echo "   ✓ Generation completed!"
echo ""

# 5. Download result
echo "5. Downloading generated audio..."
OUTPUT_FILE="test_generated_$JOB_ID.wav"
curl -s $API_URL/api/download/$JOB_ID -o $OUTPUT_FILE

if [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(ls -lh $OUTPUT_FILE | awk '{print $5}')
    echo "   ✓ Downloaded: $OUTPUT_FILE ($FILE_SIZE)"
    
    # Check audio properties
    DURATION=$(ffprobe -i $OUTPUT_FILE -show_entries format=duration -v quiet -of csv="p=0")
    echo "   Duration: $DURATION seconds"
    echo "   ✓ Audio file valid"
else
    echo "   ✗ Download failed"
    exit 1
fi
echo ""

# 6. Cleanup
echo "6. Cleaning up..."
curl -s -X POST $API_URL/api/cleanup/$JOB_ID > /dev/null
echo "   ✓ Cleanup complete"
echo ""

echo "=================================================="
echo "✓ ALL TESTS PASSED!"
echo "=================================================="
echo ""
echo "Generated file: $OUTPUT_FILE"
echo "Play with: open $OUTPUT_FILE"
echo ""
