#!/usr/bin/env bash
# Setup required Ollama models for LogMind AI
# Usage: ./scripts/setup_models.sh

set -e

echo "=== LogMind AI - Model Setup ==="

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "ERROR: Ollama is not running. Please start Ollama first:"
    echo "  ollama serve"
    exit 1
fi

echo "✓ Ollama is running"

# Pull embedding model
echo ""
echo "Pulling embedding model: nomic-embed-text..."
ollama pull nomic-embed-text

# Pull chat model
echo ""
echo "Pulling chat model: nemotron-mini..."
ollama pull nemotron-mini

echo ""
echo "=== Model Setup Complete ==="
echo ""
echo "Available models:"
ollama list

echo ""
echo "You can now run LogMind AI with:"
echo "  make run"
