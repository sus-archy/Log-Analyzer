# LogMind AI - Documentation

Welcome to the LogMind AI documentation.

## Quick Links

| Document | Description |
|----------|-------------|
| [Project Structure](PROJECT_STRUCTURE.md) | Project organization and layout |
| [API Documentation](DOCUMENTATION.md) | REST API reference |
| [ML Documentation](ML_DOCUMENTATION.md) | Machine learning module guide |
| [Architecture](ARCHITECTURE.md) | System design and architecture |
| [Roadmap](ROADMAP.md) | Feature roadmap and plans |

## Getting Started

See the main [README](../README.md) for quick start instructions.

## Overview

LogMind AI is a local-first log observability platform featuring:

- **Log Ingestion** - Automatic parsing of various log formats
- **Template Mining** - Drain algorithm for pattern extraction  
- **Semantic Search** - FAISS-powered vector similarity search
- **AI Chat** - LLM-powered log analysis with citations
- **ML Analytics** - Anomaly detection, classification, threat detection
- **Predictive Analytics** - Time-series forecasting

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Next.js Frontend                       │
└─────────────────────────────┬───────────────────────────────┘
                              │ HTTP/REST
┌─────────────────────────────▼───────────────────────────────┐
│                     FastAPI Backend                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Routes: /ingest /logs /templates /chat /ml         │   │
│  └──────────────────────────┬──────────────────────────┘   │
│  ┌──────────┬───────────────┼───────────────┬──────────┐   │
│  │ Services │    Storage    │   ML Models   │   LLM    │   │
│  └──────────┴───────────────┼───────────────┴──────────┘   │
│                             │                               │
│  ┌──────────────────────────┴──────────────────────────┐   │
│  │           SQLite + FAISS Vector Index               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    Ollama (Local LLM)                       │
│             nemotron-mini + nomic-embed-text                │
└─────────────────────────────────────────────────────────────┘
```
