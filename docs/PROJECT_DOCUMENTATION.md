# LogMind AI - Complete Project Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Installation & Setup](#3-installation--setup)
4. [Configuration](#4-configuration)
5. [Features Guide](#5-features-guide)
6. [API Reference](#6-api-reference)
7. [Database Schema](#7-database-schema)
8. [Machine Learning System](#8-machine-learning-system)
9. [Security](#9-security)
10. [Troubleshooting](#10-troubleshooting)
11. [Performance Tuning](#11-performance-tuning)

---

## 1. Project Overview

### 1.1 What is LogMind AI?

LogMind AI is a **local-first, AI-powered log observability platform** designed to help developers, DevOps engineers, and security analysts understand, search, and diagnose issues in their log data without relying on cloud services.

### 1.2 Key Features

| Feature | Description |
|---------|-------------|
| **Smart Log Ingestion** | Parse and ingest logs from various formats (JSONL, CSV, plain text) |
| **Template Mining** | Automatically extract log patterns using the Drain3 algorithm |
| **AI Chat** | Natural language interface to query and analyze logs |
| **Semantic Search** | Find similar logs using vector embeddings (FAISS) |
| **Anomaly Detection** | ML-powered detection of unusual log patterns |
| **Security Threat Detection** | Identify potential security issues (brute force, injection, etc.) |
| **Log Classification** | Automatically categorize logs by type and severity |
| **Predictive Analytics** | Forecast system health trends |

### 1.3 Technology Stack

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                              │
│  Next.js 14 │ React 18 │ TypeScript │ Tailwind CSS │ Zustand │
├─────────────────────────────────────────────────────────────┤
│                        BACKEND                               │
│  FastAPI │ Python 3.11+ │ Pydantic │ aiosqlite │ Uvicorn    │
├─────────────────────────────────────────────────────────────┤
│                         AI/ML                                │
│  Ollama (LLM) │ FAISS (Vectors) │ NumPy │ Drain3 (Parsing)  │
├─────────────────────────────────────────────────────────────┤
│                        STORAGE                               │
│  SQLite (WAL mode) │ File-based models │ FAISS index        │
└─────────────────────────────────────────────────────────────┘
```

### 1.4 Why Local-First?

- **Privacy**: Your logs never leave your machine
- **Speed**: No network latency for queries
- **Cost**: No cloud bills or API costs
- **Control**: Full control over your data and models

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                              USER                                    │
│                        (Browser: localhost:3000)                     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     NEXT.JS FRONTEND (Port 3000)                     │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ LogExplorer  │  │  AIChat      │  │ AnomalyView  │               │
│  │ (browse logs)│  │ (AI queries) │  │ (ML insights)│               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
│         │                 │                 │                        │
│  ┌──────┴─────────────────┴─────────────────┴──────┐                │
│  │              API Client (api.ts)                 │                │
│  │  • Retry with exponential backoff               │                │
│  │  • Request deduplication                         │                │
│  │  • Timeout handling                              │                │
│  └───────────────────────┬─────────────────────────┘                │
└──────────────────────────┼──────────────────────────────────────────┘
                           │ HTTP REST (via /api proxy or direct)
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND (Port 8000)                        │
│                                                                      │
│  ┌────────────────────── MIDDLEWARE ──────────────────────────┐     │
│  │ CORS │ Rate Limiting │ Error Handlers │ Request Logging    │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  ┌────────────────────── ROUTES ──────────────────────────────┐     │
│  │ /health │ /ingest │ /logs │ /templates │ /chat │ /ml       │     │
│  └────────────────────────────┬───────────────────────────────┘     │
│                               │                                      │
│  ┌────────────────────────────▼───────────────────────────────┐     │
│  │                      SERVICES                               │     │
│  │ IngestService │ QueryService │ ChatService │ MLServices    │     │
│  └────────────────────────────┬───────────────────────────────┘     │
│                               │                                      │
│  ┌────────────────────────────▼───────────────────────────────┐     │
│  │                   STORAGE LAYER                             │     │
│  │ LogsRepo │ TemplatesRepo │ VectorsRepo │ FAISS Index       │     │
│  └────────────────────────────────────────────────────────────┘     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
     ┌──────────┐   ┌──────────┐   ┌──────────────┐
     │ SQLite   │   │ FAISS    │   │ Ollama       │
     │ Database │   │ Index    │   │ (Port 11434) │
     └──────────┘   └──────────┘   └──────────────┘
```

### 2.2 Data Flow Diagrams

#### Log Ingestion Flow

```
                    ┌─────────────────┐
                    │   Log Files     │
                    │ (JSONL/CSV/TXT) │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  File Discovery │
                    │  (scan folders) │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   File Parser   │
                    │ (detect format) │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Drain3 Mining  │
                    │ (extract template│
                    │  & parameters)  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Batch Insert   │
                    │ (1000 at a time)│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Database     │
                    │ (logs_stream +  │
                    │  log_templates) │
                    └─────────────────┘
```

#### AI Chat Flow

```
┌──────────────┐
│ User Question│
└──────┬───────┘
       │
       ▼
┌──────────────────┐     ┌─────────────────────────────────┐
│ Is Log Search?   │─Yes─▶│ Search Database by Service/Term │
│ (detect intent)  │      │ Return formatted log results    │
└──────┬───────────┘      └─────────────────────────────────┘
       │ No
       ▼
┌──────────────────┐
│ ML Analysis      │
│ (Anomaly, Class, │
│  Security)       │
└──────┬───────────┘
       │
       ▼
┌───────────────┐      ┌────────────────────────────────┐
│ Fast Mode?    │─Yes─>│ Generate response from         │
│               │      │ built-in knowledge base        │
└──────┬────────┘      └────────────────────────────────┘
       │ No (LLM Mode)
       ▼
┌──────────────────┐
│ Build LLM Prompt │
│ (include ML      │
│  analysis + logs)│
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Ollama LLM       │
│ (qwen2.5:3b)     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Format Response  │
│ (markdown + logs)│
└──────────────────┘
```

### 2.3 Directory Structure

```
Log_Analyzer/
├── apps/
│   ├── api/                      # FastAPI Backend
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py           # Application entry point
│   │   │   ├── core/             # Core infrastructure
│   │   │   │   ├── config.py     # Pydantic settings
│   │   │   │   ├── security.py   # JWT authentication
│   │   │   │   ├── cache.py      # In-memory TTL cache
│   │   │   │   └── rate_limit.py # Rate limiting
│   │   │   ├── routes/           # API endpoints
│   │   │   │   ├── health.py     # Health check
│   │   │   │   ├── ingest.py     # Log ingestion
│   │   │   │   ├── logs.py       # Log queries
│   │   │   │   ├── templates.py  # Template patterns
│   │   │   │   ├── chat.py       # AI chat
│   │   │   │   ├── ml.py         # ML endpoints
│   │   │   │   └── ...
│   │   │   ├── services/         # Business logic
│   │   │   │   ├── ingest.py     # Ingestion service
│   │   │   │   ├── query.py      # Query service
│   │   │   │   ├── chat.py       # Chat service
│   │   │   │   └── ...
│   │   │   ├── storage/          # Data access layer
│   │   │   │   ├── db.py         # Database connection
│   │   │   │   ├── logs_repo.py  # Log repository
│   │   │   │   ├── templates_repo.py
│   │   │   │   └── migrations.py # Schema migrations
│   │   │   ├── ml/               # Machine learning
│   │   │   │   ├── anomaly_detector.py
│   │   │   │   ├── log_classifier.py
│   │   │   │   ├── security_threat_detector.py
│   │   │   │   ├── predictive_analytics.py
│   │   │   │   └── training_pipeline.py
│   │   │   ├── parsers/          # Log parsing
│   │   │   │   ├── drain_miner.py
│   │   │   │   ├── text_parser.py
│   │   │   │   └── normalize.py
│   │   │   ├── vector/           # Vector search
│   │   │   │   ├── faiss_index.py
│   │   │   │   └── vector_codec.py
│   │   │   └── llm/              # LLM integration
│   │   │       └── ollama_client.py
│   │   ├── data/                 # Data storage
│   │   ├── tests/                # Unit tests
│   │   └── requirements.txt      # Python dependencies
│   │
│   └── ui/                       # Next.js Frontend
│       ├── src/
│       │   ├── app/              # Next.js app router
│       │   │   ├── page.tsx      # Home page
│       │   │   ├── layout.tsx    # Root layout
│       │   │   └── api/          # API routes
│       │   ├── components/       # React components
│       │   │   ├── AIChat.tsx
│       │   │   ├── LogExplorer.tsx
│       │   │   ├── AnomalyDashboard.tsx
│       │   │   └── ...
│       │   ├── lib/              # Utilities
│       │   │   └── api.ts        # API client
│       │   └── types.ts          # TypeScript types
│       ├── package.json
│       └── next.config.js
│
├── data/                         # Shared data directory
│   ├── logmind.sqlite           # Main database
│   ├── faiss.index              # Vector index
│   └── models/                  # ML model files
│
├── Logs/                        # Sample log files
│   └── loghub/                  # LogHub datasets
│
├── scripts/                     # Utility scripts
│   ├── ingest_folder.py        # Batch ingestion
│   ├── setup_models.sh         # Download Ollama models
│   └── smoke.sh                # Smoke tests
│
├── docs/                        # Documentation
├── Makefile                     # Build commands
├── start.sh                     # Start script
└── stop.sh                      # Stop script
```

---

## 3. Installation & Setup

### 3.1 Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.11+ | Backend runtime |
| Node.js | 18+ | Frontend runtime |
| Ollama | Latest | Local LLM inference |
| SQLite | 3.35+ | Database |

### 3.2 Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd Log_Analyzer

# 2. Install Python dependencies
cd apps/api
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# 3. Install Node.js dependencies
cd ../ui
npm install

# 4. Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

# 5. Start the application
cd ../..
make dev
# OR manually:
# Terminal 1: cd apps/api && uvicorn app.main:app --reload --port 8000
# Terminal 2: cd apps/ui && npm run dev
```

### 3.3 Verify Installation

```bash
# Check API health
curl http://localhost:8000/health
# Expected: {"status":"ok"}

# Check frontend
open http://localhost:3000

# Check Ollama
curl http://localhost:11434/api/tags
```

---

## 4. Configuration

### 4.1 Environment Variables

Create a `.env` file in `apps/api/`:

```bash
# Database
DATABASE_PATH=../../data/logmind.sqlite

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_EMBED_MODEL=nomic-embed-text

# Security (optional)
AUTH_ENABLED=false
JWT_SECRET=your-secret-key-change-in-production
ADMIN_USERNAME=admin
ADMIN_PASSWORD=changeme

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
```

### 4.2 Pydantic Settings

All configuration is managed via `apps/api/app/core/config.py`:

```python
class Settings(BaseSettings):
    # API
    api_title: str = "LogMind AI"
    api_version: str = "1.0.0"
    
    # Database
    database_path: str = "../../data/logmind.sqlite"
    db_pool_size: int = 5
    
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout: float = 60.0
    
    # ML
    models_path: str = "../../data/models"
    anomaly_threshold: float = 0.5
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 100
```

---

## 5. Features Guide

### 5.1 Log Ingestion

#### Via UI (File Upload)
1. Navigate to the Upload section
2. Drag and drop log files or click to browse
3. Supported formats: `.log`, `.txt`, `.jsonl`, `.csv`
4. Files are processed with Drain3 template mining

#### Via API
```bash
# Single event
curl -X POST http://localhost:8000/ingest/event \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-01-15T10:30:00Z",
    "service_name": "web-server",
    "severity": 2,
    "body": "Connection timeout from 192.168.1.100"
  }'

# File upload
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@/path/to/logs.jsonl"
```

#### Via Script
```bash
# Ingest entire folder
python scripts/ingest_folder.py --path ./Logs/loghub
```

### 5.2 Log Exploration

The Log Explorer provides:
- **Time range filtering**: Preset ranges (1h, 6h, 24h, 7d, 30d, all)
- **Service filtering**: Filter by service name
- **Severity filtering**: Filter by severity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Template filtering**: Click on a template to see all matching logs
- **Full-text search**: Search in log messages
- **Pagination**: Navigate through large result sets

### 5.3 AI Chat

Two modes available:

#### Fast Mode (Default)
- **Instant responses** using built-in knowledge base
- **Log search**: "Show me error logs from apache"
- **Security Q&A**: "How do I prevent SQL injection?"
- **Diagnostic help**: "What causes timeout errors?"

#### LLM Mode
- **AI-powered** responses using Ollama
- **Context-aware** analysis with ML model results
- **Longer response time** (15-60 seconds)
- **More conversational** and detailed explanations

#### Example Queries
```
# Log Search
"Show me authentication failures"
"Find logs from healthapp"
"Show me error logs from the last hour"

# Security Questions
"Is this a SQL injection: SELECT * FROM users WHERE id='1' OR '1'='1'"
"How do I prevent XSS attacks?"

# Diagnostics
"Why is my database timing out?"
"What does this error mean: Connection refused"
"Analyze this log: [paste log message]"
```

### 5.4 Semantic Search

Find similar logs using vector embeddings:
1. Navigate to Semantic Search
2. Enter a log message or description
3. See similar logs ranked by similarity score
4. Uses FAISS for fast nearest-neighbor search

### 5.5 Anomaly Detection

The ML-powered anomaly dashboard shows:
- **Anomaly score**: 0-1 scale (higher = more unusual)
- **Anomaly type**: Statistical outlier, rare pattern, temporal anomaly
- **Contributing factors**: What made this log unusual
- **Time-series visualization**: Anomaly trends over time

### 5.6 Security Analysis

Detect potential security threats:
- **Brute force attacks**: Multiple failed logins
- **Injection attempts**: SQL, command, XSS
- **Reconnaissance**: Port scanning, enumeration
- **Attack patterns**: Known malicious signatures

---

## 6. API Reference

### 6.1 Health & Status

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

#### GET /ml/train/status
Get ML model training status.

**Response:**
```json
{
  "models_exist": {
    "anomaly_detector": true,
    "log_classifier": true,
    "security_detector": true
  },
  "last_trained": "2024-01-15T10:30:00Z"
}
```

### 6.2 Log Ingestion

#### POST /ingest/event
Ingest a single log event.

**Request Body:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "service_name": "web-server",
  "severity": 2,
  "body": "User login successful for user@example.com",
  "trace_id": "abc123",
  "attributes": {"user_id": "123"}
}
```

#### POST /ingest/batch
Ingest multiple events.

**Request Body:**
```json
{
  "events": [
    {"timestamp": "...", "service_name": "...", "body": "..."},
    {"timestamp": "...", "service_name": "...", "body": "..."}
  ]
}
```

#### POST /ingest/upload
Upload a log file.

**Form Data:**
- `file`: The log file (multipart/form-data)

### 6.3 Log Queries

#### GET /logs
Query logs with filters.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| from | string | Start time (ISO 8601) |
| to | string | End time (ISO 8601) |
| service_name | string | Filter by service |
| severity_min | int | Minimum severity (0-5) |
| template_hash | string | Filter by template |
| limit | int | Max results (default 100) |
| offset | int | Pagination offset |

**Example:**
```bash
curl "http://localhost:8000/logs?from=2024-01-01T00:00:00Z&to=2024-01-15T23:59:59Z&service_name=apache&limit=50"
```

#### GET /logs/services/list
Get list of all services.

**Response:**
```json
{
  "services": ["apache", "nginx", "mysql", "app-server"]
}
```

### 6.4 Templates

#### GET /templates
Query log templates.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| service_name | string | Filter by service |
| from | string | Start time |
| to | string | End time |
| limit | int | Max results |

#### GET /templates/{hash}
Get template details with example logs.

### 6.5 AI Chat

#### POST /ml/chat
Send a message to the AI chat.

**Request Body:**
```json
{
  "message": "Show me error logs from apache",
  "fast_mode": true,
  "use_llm": false,
  "context": ["previous message 1", "previous message 2"]
}
```

**Response:**
```json
{
  "response": "## 🔍 Log Search Results\n...",
  "analysis": {...},
  "model_status": {
    "anomaly_detector": true,
    "log_classifier": true,
    "security_detector": true
  },
  "suggestions": ["Why is this unusual?", "How to fix?"]
}
```

### 6.6 ML Endpoints

#### POST /ml/train
Train ML models.

**Request Body:**
```json
{
  "max_logs_per_source": 2000,
  "train_ratio": 0.8,
  "force_retrain": false
}
```

#### POST /ml/anomaly/detect
Detect anomalies in logs.

#### POST /ml/classify
Classify logs by category.

#### POST /ml/security/detect
Detect security threats.

---

## 7. Database Schema

### 7.1 Schema Version: 2

```sql
-- Template patterns extracted from logs
CREATE TABLE log_templates (
    tenant_id TEXT DEFAULT 'default',
    service_name TEXT NOT NULL,
    template_hash INTEGER NOT NULL,      -- xxhash64
    template_text TEXT NOT NULL,
    first_seen_utc TEXT,
    last_seen_utc TEXT,
    embedding_state TEXT DEFAULT 'none', -- none|pending|done
    embedding_model TEXT,
    PRIMARY KEY (tenant_id, service_name, template_hash)
);

-- Individual log entries
CREATE TABLE logs_stream (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT DEFAULT 'default',
    service_name TEXT NOT NULL,
    environment TEXT DEFAULT 'production',
    timestamp_utc TEXT NOT NULL,
    ingest_timestamp_utc TEXT,
    severity INTEGER DEFAULT 1,          -- 0=DEBUG to 4=CRITICAL
    host TEXT,
    template_hash INTEGER,
    parameters_json TEXT,                -- Extracted variables
    trace_id TEXT,
    span_id TEXT,
    attributes_json TEXT,
    body_raw TEXT NOT NULL
);

-- Vector embeddings for semantic search
CREATE TABLE template_vectors (
    tenant_id TEXT DEFAULT 'default',
    service_name TEXT NOT NULL,
    template_hash INTEGER NOT NULL,
    faiss_id INTEGER NOT NULL,
    vector_b64 TEXT,                     -- Base64 encoded vector
    PRIMARY KEY (tenant_id, service_name, template_hash)
);

-- Track ingested files
CREATE TABLE ingested_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_size INTEGER,
    file_hash TEXT,                      -- For duplicate detection
    status TEXT DEFAULT 'pending',
    lines_processed INTEGER DEFAULT 0,
    events_inserted INTEGER DEFAULT 0,
    created_at TEXT,
    completed_at TEXT
);

-- Schema version tracking
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY
);
```

### 7.2 Indexes

```sql
-- Performance indexes
CREATE INDEX idx_logs_timestamp ON logs_stream(timestamp_utc);
CREATE INDEX idx_logs_severity ON logs_stream(severity);
CREATE INDEX idx_logs_tenant_time ON logs_stream(tenant_id, timestamp_utc);
CREATE INDEX idx_logs_service_time ON logs_stream(service_name, timestamp_utc);
CREATE INDEX idx_logs_template ON logs_stream(template_hash);
```

### 7.3 Severity Levels

| Value | Name | Description |
|-------|------|-------------|
| 0 | DEBUG | Detailed diagnostic information |
| 1 | INFO | Normal operational messages |
| 2 | WARNING | Warning conditions |
| 3 | ERROR | Error conditions |
| 4 | CRITICAL | Critical/fatal errors |

---

## 8. Machine Learning System

### 8.1 Overview

LogMind AI uses custom ML models implemented in pure NumPy (no sklearn dependency):

| Model | Purpose | Algorithm |
|-------|---------|-----------|
| AnomalyDetector | Find unusual logs | Isolation Forest + Statistical |
| LogClassifier | Categorize logs | TF-IDF + Naive Bayes |
| SecurityThreatDetector | Detect attacks | Markov Chains + Pattern Matching |
| PredictiveAnalytics | Forecast trends | Holt-Winters |

### 8.2 Training Pipeline

```bash
# Train all models
curl -X POST http://localhost:8000/ml/train \
  -H "Content-Type: application/json" \
  -d '{"force_retrain": true}'
```

Training process:
1. Load logs from database or Logs/loghub folder
2. Extract features (text, timing, frequency, severity)
3. Train each model independently
4. Evaluate on held-out test set
5. Save models to `data/models/`

### 8.3 Model Files

```
data/models/
├── anomaly_detector.pkl
├── log_classifier.pkl
├── security_detector.pkl
└── predictive_analytics.pkl
```

### 8.4 Anomaly Detection

**Features extracted:**
- Log message length
- Word count
- Special character frequency
- Numeric token ratio
- Time of day
- Inter-arrival time

**Detection methods:**
1. **Isolation Forest**: Tree-based outlier detection
2. **Statistical**: Z-score and IQR bounds
3. **DBSCAN**: Density-based clustering

### 8.5 Security Threat Detection

**Threat types detected:**
- Brute force attacks (multiple failed logins)
- SQL injection patterns
- Command injection
- XSS attempts
- Path traversal
- Reconnaissance/scanning

---

## 9. Security

### 9.1 Authentication (Optional)

JWT-based authentication can be enabled:

```bash
# In .env
AUTH_ENABLED=true
JWT_SECRET=your-secure-random-string
```

### 9.2 Rate Limiting

Per-endpoint rate limits protect against abuse:

| Endpoint | Limit |
|----------|-------|
| /ingest/* | 20/min |
| /logs | 100/min |
| /chat | 10/min |
| /ml/train | 5/min |

### 9.3 Input Validation

All inputs are validated using Pydantic models:
- Type checking
- Length limits
- Format validation
- HTML/script tag sanitization

### 9.4 Security Best Practices

1. **Change default credentials** if auth is enabled
2. **Use HTTPS** in production (via reverse proxy)
3. **Restrict CORS origins** to your domains
4. **Keep Ollama local** (don't expose to internet)
5. **Regular backups** of SQLite database

---

## 10. Troubleshooting

### 10.1 Common Issues

#### API Server Won't Start
```bash
# Check if port is in use
lsof -i :8000

# Check Python environment
which python
python --version

# Check dependencies
pip install -r requirements.txt
```

#### Ollama Not Responding
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Pull required models
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

#### Frontend Build Errors
```bash
# Clear cache and reinstall
cd apps/ui
rm -rf node_modules .next
npm install
npm run dev
```

#### Database Locked
```bash
# SQLite WAL cleanup
cd data
sqlite3 logmind.sqlite "PRAGMA wal_checkpoint(TRUNCATE);"
```

### 10.2 Log Locations

| Component | Log Location |
|-----------|--------------|
| API | stdout (uvicorn) |
| Frontend | Browser console |
| Ollama | ~/.ollama/logs |

### 10.3 Debug Mode

```bash
# Run API in debug mode
cd apps/api
uvicorn app.main:app --reload --log-level debug

# Frontend verbose mode
cd apps/ui
npm run dev -- --verbose
```

---

## 11. Performance Tuning

### 11.1 Database Optimization

```sql
-- Analyze tables for query planner
ANALYZE;

-- Vacuum to reclaim space
VACUUM;

-- Check index usage
EXPLAIN QUERY PLAN SELECT * FROM logs_stream WHERE timestamp_utc > '2024-01-01';
```

### 11.2 Memory Settings

In `config.py`:
```python
# Increase connection pool
db_pool_size: int = 10

# Increase cache size
cache_max_items: int = 10000
cache_ttl_seconds: int = 600
```

### 11.3 Batch Sizes

For large ingestion jobs:
```python
# In ingest.py
BATCH_SIZE = 5000  # Increase for faster ingestion
COMMIT_INTERVAL = 10000
```

### 11.4 LLM Performance

- Use smaller models for faster responses: `qwen2.5:1.5b`
- Increase Ollama memory: `OLLAMA_NUM_PARALLEL=4`
- Use Fast Mode for instant responses

---

## Appendix A: Makefile Commands

```makefile
make dev        # Start both API and UI in development mode
make api        # Start API only
make ui         # Start UI only
make test       # Run all tests
make lint       # Run linters
make format     # Format code
make clean      # Clean build artifacts
make ingest     # Run ingestion script
make train      # Train ML models
```

---

## Appendix B: Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Enter | Send chat message |
| Ctrl+K | Focus search |
| Ctrl+/ | Toggle help |
| Esc | Close modal |

---

*Documentation Version: 1.0.0*
*Last Updated: December 28, 2025*
