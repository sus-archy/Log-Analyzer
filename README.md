# LogMind AI

A local-first log observability platform with AI-powered insights.

## Features

- **Log Ingestion**: Automatically ingest logs from local folders (JSONL, CSV, text formats)
- **Template Mining**: Drain-based algorithm extracts reusable log templates with parameters
- **Semantic Search**: FAISS-powered vector similarity search using Ollama embeddings
- **AI Chat**: Ask questions about your logs with grounded, citation-backed answers
- **ML Analytics**: Real machine learning for anomaly detection, classification, and threat detection
- **Predictive Analytics**: Time-series forecasting for system health prediction
- **Web UI**: Modern Next.js interface with Explorer, Templates, and Chat views

## Documentation

📚 See the [docs/](docs/) folder for detailed documentation:

| Document | Description |
|----------|-------------|
| [Project Documentation](PROJECT_DOCUMENTATION.md) | Full Project Documentation |
| [Code Documentation](Code_Dodumentation.md) | Code Explanation |
| [ML Documentation](ML_DOCUMENTATION.md) | Machine learning module guide |
| [Architecture](ARCHITECTURE.md) | System design and architecture |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Next.js UI                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Log Explorer│  │  Templates  │  │      Chat Panel         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP
┌────────────────────────────▼────────────────────────────────────┐
│                      FastAPI Backend                            │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌───────────────────┐   │
│  │ Ingest  │ │  Logs    │ │ Templates │ │ Semantic + Chat   │   │
│  │ Service │ │  Service │ │  Service  │ │ Services          │   │
│  └────┬────┘ └────┬─────┘ └─────┬─────┘ └──────────┬────────┘   │
│       │           │             │                  │            │
│  ┌────▼───────────▼─────────────▼──────────────────▼─────────┐  │
│  │                     Storage Layer                         │  │
│  │  ┌───────────────────────────────────────────────────────┐│  │
│  │  │                 SQLite Database                       ││  │
│  │  │  • log_templates  • logs_stream  • template_vectors   ││  │
│  │  └───────────────────────────────────────────────────────┘│  │
│  │  ┌───────────────────────────────────────────────────────┐│  │
│  │  │                 FAISS Index                           ││  │
│  │  │  • Vector similarity search (IndexFlatIP)             ││  │
│  │  └───────────────────────────────────────────────────────┘│  │
│  └───────────────────────────────────────────────────────────┘  │ 
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                        Ollama (Local)                           │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐  │
│  │  nemotron-mini      │  │  nomic-embed-text                │  │
│  │  (Chat Model)       │  │  (Embedding Model)               │  │
│  └─────────────────────┘  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

1. **Python 3.11+**
2. **Node.js 18+**
3. **Ollama** - Install from https://ollama.ai

### Setup

```bash
# 1. Install dependencies
make install

# 2. Start Ollama and pull required models
ollama serve  # In a separate terminal
make setup-models

# 3. Create data directory
make init

# 4. Run the application
make run
```

The UI will be available at http://localhost:3000

### Alternative: Manual Setup

```bash
# Backend
cd apps/api
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (in another terminal)
cd apps/ui
npm install
npm run dev
```

## Configuration

Environment variables (set in `.env` or shell):

| Variable | Default | Description |
|----------|---------|-------------|
| `LOGS_FOLDER` | `./Logs` | Path to logs folder for ingestion |
| `SQLITE_PATH` | `./data/logmind.sqlite` | SQLite database path |
| `FAISS_INDEX_PATH` | `./data/faiss_index.bin` | FAISS index path |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:3b` | Chat model name |
| `EMBED_DIM` | `768` | Embedding dimension |
| `AUTH_ENABLED` | `false` | Enable JWT authentication |
| `SECRET_KEY` | (random) | JWT signing key |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed CORS origins |
| `RATE_LIMIT_CHAT` | `20/minute` | Rate limit for chat endpoint |
| `DB_POOL_SIZE` | `5` | Database connection pool size |

## API Endpoints

### Health
- `GET /health` - Health check (includes Ollama status)

### Authentication (when enabled)
- `POST /auth/token` - Get JWT access token
- `GET /auth/me` - Get current user info
- `GET /auth/status` - Check auth configuration

### Ingest
- `POST /ingest` - Ingest logs from configured folder

### Logs
- `GET /logs` - Query logs with filters
- `GET /logs/services` - List distinct services
- `GET /logs/stats/quick` - Quick statistics

### Templates
- `GET /templates/top` - Get top templates by occurrence
- `GET /templates/{hash}` - Get template details
- `POST /templates/embed` - Embed all pending templates

### Semantic Search
- `POST /semantic/search` - Search templates by semantic similarity

### Chat (rate limited)
- `POST /chat` - Ask questions about logs with AI

### Metrics
- `GET /metrics/performance` - Performance metrics
- `GET /metrics/security` - Security analysis
- `GET /metrics/services/health` - Service health status

## Project Structure

```
├── apps/
│   ├── api/                    # FastAPI backend
│   │   ├── app/
│   │   │   ├── core/           # Config, logging, security, rate limiting
│   │   │   ├── llm/            # Ollama client
│   │   │   ├── parsers/        # Log parsing, Drain miner
│   │   │   ├── routes/         # API routes (with auth)
│   │   │   ├── schemas/        # Pydantic models (with validation)
│   │   │   ├── services/       # Business logic (metrics, security)
│   │   │   ├── storage/        # Database (with connection pool), repos
│   │   │   └── vector/         # FAISS index
│   │   ├── tests/              # Unit tests
│   │   └── requirements.txt
│   └── ui/                     # Next.js frontend
│       ├── src/
│       │   ├── app/            # Pages
│       │   ├── components/     # React components (with ErrorBoundary)
│       │   ├── lib/            # API client, Zustand store
│       │   └── types.ts        # TypeScript types
│       └── package.json
├── Logs/                       # Log files for ingestion
├── data/                       # SQLite DB, FAISS index
├── scripts/                    # Setup and utility scripts
└── Makefile
```

## Development

```bash
# Run in development mode with hot reload
make dev

# Run tests
make test

# Run smoke tests (requires running backend)
make smoke

# Lint code
make lint
```

## Usage Guide

### 1. Ingest Logs

Place your log files in the `Logs/` folder (supports `.log`, `.txt`, `.jsonl`, `.csv`). Then:

```bash
# Via API
curl -X POST http://localhost:8000/ingest

# Or use the "Ingest Logs" button in the UI
```

### 2. Explore Logs

Open the **Explorer** tab to browse logs:
- Filter by service, severity, time range
- Click a row to expand and see details
- Search within results

### 3. View Templates

Open the **Templates** tab to see extracted patterns:
- Templates group similar logs together
- Parameters are shown as `<*>` placeholders
- Click a template to see recent occurrences

### 4. Embed Templates

Before using semantic search or chat, embed the templates:

```bash
# Via API
curl -X POST http://localhost:8000/templates/embed

# Or use the "Embed Templates" button in the UI
```

### 5. Chat with Logs

Open the **Chat** tab and ask questions like:
- "What errors occurred in the last hour?"
- "Why are there connection failures?"
- "Summarize the issues for this service"
- "Show me logs related to DDoS attack"

The AI provides grounded answers with citations to specific templates and logs.
