#!/bin/bash
# Quick deployment test script for MusicLab
# Tests Docker setup and verifies all services are healthy

set -e  # Exit on error

echo "🚀 MusicLab Deployment Test Script"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found${NC}"
    echo "Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✓ Docker installed${NC}"

if ! docker compose version &> /dev/null; then
    echo -e "${RED}✗ Docker Compose not found${NC}"
    echo "Install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose installed${NC}"

# Check for .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠ .env file not found${NC}"
    echo "Creating from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${YELLOW}! Please edit .env and add your GROQ_API_KEY${NC}"
        exit 1
    else
        echo -e "${RED}✗ .env.example not found${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓ .env file exists${NC}"

# Check for GROQ_API_KEY
if ! grep -q "^GROQ_API_KEY=gsk_" .env; then
    echo -e "${YELLOW}⚠ GROQ_API_KEY not configured in .env${NC}"
    echo "Please add your Groq API key to .env file"
    echo "Get one from: https://console.groq.com"
    exit 1
fi
echo -e "${GREEN}✓ GROQ_API_KEY configured${NC}"

echo ""
echo "🔨 Building Docker images..."
docker compose build

echo ""
echo "🚀 Starting services..."
docker compose up -d

echo ""
echo "⏳ Waiting for services to start (60s for model loading)..."
sleep 60

echo ""
echo "🔍 Checking container status..."
if ! docker ps | grep -q musiclab-backend; then
    echo -e "${RED}✗ Backend container not running${NC}"
    docker compose logs backend
    exit 1
fi
echo -e "${GREEN}✓ Backend container running${NC}"

if ! docker ps | grep -q musiclab-frontend; then
    echo -e "${RED}✗ Frontend container not running${NC}"
    docker compose logs frontend
    exit 1
fi
echo -e "${GREEN}✓ Frontend container running${NC}"

echo ""
echo "🏥 Checking health status..."

# Check backend health
if ! curl -f http://localhost:8001/health &> /dev/null; then
    echo -e "${RED}✗ Backend health check failed${NC}"
    echo "Backend logs:"
    docker compose logs --tail=50 backend
    exit 1
fi
echo -e "${GREEN}✓ Backend healthy${NC}"

# Check frontend
if ! curl -f -I http://localhost &> /dev/null; then
    echo -e "${RED}✗ Frontend not responding${NC}"
    echo "Frontend logs:"
    docker compose logs --tail=50 frontend
    exit 1
fi
echo -e "${GREEN}✓ Frontend responding${NC}"

echo ""
echo "✅ All tests passed!"
echo ""
echo "🌐 Access points:"
echo "  Frontend:    http://localhost"
echo "  Frontend:    http://localhost:3000 (alternative)"
echo "  Backend API: http://localhost:8001/docs"
echo "  Health:      http://localhost:8001/health"
echo ""
echo "📊 View logs:"
echo "  All:         docker compose logs -f"
echo "  Backend:     docker compose logs -f backend"
echo "  Frontend:    docker compose logs -f frontend"
echo ""
echo "🛑 Stop services:"
echo "  docker compose down"
echo ""
