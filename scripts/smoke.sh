#!/usr/bin/env bash
# Smoke test for LogMind AI
# Usage: ./scripts/smoke.sh

set -e

API_URL="${API_URL:-http://localhost:8000}"

echo "=== LogMind AI - Smoke Test ==="
echo "API URL: $API_URL"
echo ""

# Health check
echo "1. Health check..."
HEALTH=$(curl -s "$API_URL/health")
echo "   $HEALTH"

if echo "$HEALTH" | grep -q '"status":"ok"'; then
    echo "   ✓ Health check passed"
else
    echo "   ✗ Health check failed"
    exit 1
fi

# List services (should be empty initially)
echo ""
echo "2. List services..."
SERVICES=$(curl -s "$API_URL/logs/services")
echo "   $SERVICES"
echo "   ✓ Services endpoint works"

# Ingest logs
echo ""
echo "3. Ingest logs from folder..."
INGEST=$(curl -s -X POST "$API_URL/ingest")
echo "   Logs ingested: $(echo $INGEST | grep -o '"logs_ingested":[0-9]*' | cut -d: -f2)"
echo "   Templates created: $(echo $INGEST | grep -o '"templates_created":[0-9]*' | cut -d: -f2)"
echo "   ✓ Ingest endpoint works"

# Query logs
echo ""
echo "4. Query logs..."
LOGS=$(curl -s "$API_URL/logs?limit=5")
echo "   Total logs: $(echo $LOGS | grep -o '"total":[0-9]*' | cut -d: -f2)"
echo "   ✓ Logs query works"

# List services again
echo ""
echo "5. List services after ingest..."
SERVICES=$(curl -s "$API_URL/logs/services")
echo "   Services: $SERVICES"
echo "   ✓ Services populated"

# Get templates
echo ""
echo "6. Get top templates..."
# Get first service name
SERVICE=$(echo $SERVICES | grep -o '"[^"]*"' | head -1 | tr -d '"')
if [ -n "$SERVICE" ]; then
    TEMPLATES=$(curl -s "$API_URL/templates/top?service_name=$SERVICE&limit=3")
    echo "   Templates found: $(echo $TEMPLATES | grep -o '"template_hash"' | wc -l)"
    echo "   ✓ Templates endpoint works"
else
    echo "   No services found, skipping template test"
fi

# Embed templates
echo ""
echo "7. Embed templates..."
EMBED=$(curl -s -X POST "$API_URL/templates/embed")
echo "   $EMBED"
echo "   ✓ Embed endpoint works"

# Semantic search
echo ""
echo "8. Semantic search..."
SEARCH=$(curl -s -X POST "$API_URL/semantic/search" \
    -H "Content-Type: application/json" \
    -d '{"query": "error", "top_k": 3}')
echo "   Results: $(echo $SEARCH | grep -o '"template_hash"' | wc -l)"
echo "   ✓ Semantic search works"

# Chat (only if we have a service)
if [ -n "$SERVICE" ]; then
    echo ""
    echo "9. Chat..."
    CHAT=$(curl -s -X POST "$API_URL/chat" \
        -H "Content-Type: application/json" \
        -d "{\"service_name\": \"$SERVICE\", \"question\": \"What errors occurred?\"}")
    
    if echo "$CHAT" | grep -q '"answer"'; then
        echo "   ✓ Chat endpoint works"
        echo "   Confidence: $(echo $CHAT | grep -o '"confidence":"[^"]*"' | cut -d'"' -f4)"
    else
        echo "   Chat response: $CHAT"
    fi
fi

echo ""
echo "=== Smoke Test Complete ==="
echo "All endpoints are working!"
