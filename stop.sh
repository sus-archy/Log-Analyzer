#!/bin/bash
# LogMind AI - Stop Script
# Usage: ./stop.sh

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🛑 Stopping LogMind AI...${NC}"

pkill -f "uvicorn.*app.main:app" 2>/dev/null && echo -e "${GREEN}✓ API server stopped${NC}" || echo -e "${YELLOW}• API server not running${NC}"
pkill -f "next.*dev" 2>/dev/null && echo -e "${GREEN}✓ UI server stopped${NC}" || echo -e "${YELLOW}• UI server not running${NC}"

echo -e "${GREEN}Done!${NC}"
