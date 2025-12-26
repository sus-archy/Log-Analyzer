#!/bin/bash
# LogMind AI - Easy Startup Script
# Usage: ./start.sh

PROJECT_DIR="/home/bug/Desktop/Log_Analyzer"
VENV_DIR="$PROJECT_DIR/.venv"
API_DIR="$PROJECT_DIR/apps/api"
UI_DIR="$PROJECT_DIR/apps/ui"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧠 LogMind AI - Starting...${NC}"
echo ""

# Kill existing processes
echo -e "${YELLOW}Stopping existing servers...${NC}"
pkill -9 -f "uvicorn.*app.main:app" 2>/dev/null || true
pkill -9 -f "next.*dev" 2>/dev/null || true
sleep 2

# Check Ollama
echo -e "${YELLOW}Checking Ollama...${NC}"
if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama is running${NC}"
else
    echo -e "${RED}✗ Ollama is not running. Please start it with: ollama serve${NC}"
    exit 1
fi

# Start API server
echo -e "${YELLOW}Starting API server...${NC}"
cd "$API_DIR"
PYTHONPATH="$API_DIR" "$VENV_DIR/bin/uvicorn" app.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/logmind-api.log 2>&1 &
API_PID=$!
sleep 4

# Check if API started
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ API server started (PID: $API_PID)${NC}"
else
    echo -e "${RED}✗ API server failed to start. Check /tmp/logmind-api.log${NC}"
    cat /tmp/logmind-api.log | tail -20
    exit 1
fi

# Start UI server
echo -e "${YELLOW}Starting UI server...${NC}"
cd "$UI_DIR"
npm run dev > /tmp/logmind-ui.log 2>&1 &
UI_PID=$!
sleep 6

# Check if UI started
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ UI server started (PID: $UI_PID)${NC}"
else
    echo -e "${YELLOW}⚠ UI server starting... (check http://localhost:3000)${NC}"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 LogMind AI is running!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${BLUE}📊 Dashboard:${NC} http://localhost:3000"
echo -e "  ${BLUE}🔌 API:${NC}       http://localhost:8000"
echo -e "  ${BLUE}📚 API Docs:${NC}  http://localhost:8000/docs"
echo ""
echo -e "  ${YELLOW}To stop:${NC} ./stop.sh"
echo ""
