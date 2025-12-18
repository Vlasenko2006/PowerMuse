#!/bin/bash
# Monitor Docker build progress

echo "=== Docker Build Monitor ==="
echo ""

# Check if build process is running
if ps aux | grep -q "[d]ocker.*compose.*build"; then
    echo "✅ Build process is RUNNING"
    echo ""
    echo "Process details:"
    ps aux | grep "[d]ocker.*compose.*build" | head -2
    echo ""
else
    echo "⚠️  No build process found"
fi

echo "=== Current Docker Images ==="
docker images | grep -E "REPOSITORY|musiclab"
echo ""

echo "=== To check again in 60 seconds, run: ==="
echo "sleep 60 && ./monitor_build.sh"
