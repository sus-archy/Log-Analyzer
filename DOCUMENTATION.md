# LogMind AI - Technical Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Directory Structure](#directory-structure)
4. [Backend Architecture](#backend-architecture-fastapi)
5. [**Cross-Domain Few-Shot Learning**](#cross-domain-few-shot-learning) ⭐ **NEW**
6. [AI Model Training Pipeline](#ai-model-training-pipeline)
7. [AI Components Deep Dive](#ai-components-deep-dive)
8. [Database Schema](#database-schema)
9. [Frontend Architecture](#frontend-architecture-nextjs)
10. [Performance & Security Metrics](#performance--security-metrics)
11. [Data Flow](#data-flow)

---

## Project Overview

**LogMind AI** is a local, AI-powered log observability and analysis platform designed for intelligent log management. It provides automated log parsing, pattern recognition, semantic search, and anomaly detection capabilities using machine learning and natural language processing.

### Key Features

- **Automated Log Parsing**: Uses the Drain algorithm to extract log templates and patterns
- **AI-Powered Semantic Search**: Natural language queries using vector embeddings
- **Cross-Domain Few-Shot Learning**: AI learns patterns from one log type to analyze others
- **Anomaly Detection**: Identifies unusual patterns using statistical analysis and AI
- **Multi-Format Support**: Handles various log formats (syslog, Apache, application logs, CSV)
- **Real-time Analysis**: Fast queries on millions of log entries
- **Interactive Dashboard**: Modern web interface for log exploration
- **Comprehensive Reports**: Performance metrics, security analysis, and PDF export

### How the AI "Learns" Your Logs

LogMind's AI doesn't train from scratch—it uses **pre-trained models** to understand your logs:

1. **Log Parsing**: Raw logs are parsed into templates (patterns)
2. **Embedding Generation**: Each template is converted into a 384-dimensional vector using `nomic-embed-text`
3. **Index Building**: Vectors are stored in FAISS for fast similarity search
4. **Query Time**: User queries are also embedded and compared to find relevant logs

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LogMind AI Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────┐ │
│  │   Frontend   │────▶│   Next.js    │────▶│      REST API Calls        | |
│  │  (Browser)   │     │   UI Server  │     │   (localhost:3000/api/*)     │ │
│  └──────────────┘     │  Port: 3000  │     └──────────────────────────────┘ │
│                       └──────────────┘                                      │
│                              │                                              │
│                              ▼                                              │
│                       ┌──────────────┐                                      │
│                       │   FastAPI    │                                      │
│                       │  API Server  │                                      │
│                       │  Port: 8000  │                                      │
│                       └──────────────┘                                      │
│                              │                                              │
│         ┌────────────────────┼────────────────────┐                         │
│         ▼                    ▼                    ▼                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │   SQLite     │     │    FAISS     │     │   Ollama     │                 │
│  │  Database    │     │ Vector Index │     │  LLM Server  │                 │
│  │ (logmind.db) │     │ (faiss.index)│     │ Port: 11434  │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | Next.js 14, React, TypeScript, Tailwind CSS | User interface |
| Backend API | FastAPI (Python 3.13) | REST API server |
| Database | SQLite with WAL mode | Log and template storage |
| Vector Store | FAISS (Facebook AI Similarity Search) | Semantic search index |
| LLM | Ollama with Qwen 2.5 (3B) | Chat and reasoning |
| Embeddings | Nomic Embed Text | Text-to-vector conversion |
| Log Parsing | Drain3 Algorithm | Template extraction |

---

## Directory Structure

```
Log_Analyzer/
├── apps/
│   ├── api/                    # Backend FastAPI application
│   │   └── app/
│   │       ├── main.py         # Application entry point
│   │       ├── core/           # Configuration and logging
│   │       ├── routes/         # API endpoints
│   │       ├── services/       # Business logic
│   │       ├── storage/        # Database repositories
│   │       ├── parsers/        # Log parsing (Drain algorithm)
│   │       ├── vector/         # FAISS vector operations
│   │       └── schemas/        # Pydantic data models
│   │
│   └── ui/                     # Frontend Next.js application
│       └── src/
│           ├── app/            # Next.js app router
│           ├── components/     # React components
│           ├── lib/            # API client and utilities
│           └── types/          # TypeScript definitions
│
├── data/                       # Runtime data (SQLite, FAISS index)
├── Logs/                       # Sample log files for ingestion
├── start.sh                    # Startup script
├── stop.sh                     # Shutdown script
└── DOCUMENTATION.md            # This file
```

---

## Backend Architecture (FastAPI)

### Core Files

#### `apps/api/app/main.py`
The application entry point that:
- Initializes the FastAPI application
- Sets up CORS middleware for frontend communication
- Registers all API routers
- Initializes database and FAISS index on startup

```python
# Key initialization sequence:
1. Initialize SQLite database with migrations
2. Load or create FAISS vector index
3. Restore embedding mappings
4. Start API server on port 8000
```

### API Routes (`apps/api/app/routes/`)

| File | Endpoints | Purpose |
|------|-----------|---------|
| `logs.py` | `/logs`, `/logs/{id}`, `/logs/services/list`, `/logs/stats/quick` | Log querying and browsing |
| `templates.py` | `/templates/top`, `/templates/{hash}` | Template analysis |
| `ingest.py` | `/ingest/file`, `/ingest/folder` | Log file ingestion |
| `semantic.py` | `/semantic_search` | AI-powered search |
| `chat.py` | `/chat`, `/chat/embed/*` | LLM chat interface |
| `anomaly.py` | `/anomaly/detect` | Anomaly detection |

### Services (`apps/api/app/services/`)

#### `ingest_service.py` - Log Ingestion Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Raw Log    |──▶│  Parser     │──▶│   Drain     │───▶│ Database   │
│   File      │    │  Detection  │    │  Algorithm  │    │   Insert    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │                  │
                          ▼                  ▼
                   ┌─────────────┐    ┌─────────────┐
                   │  Timestamp  │    │  Template   │
                   │  Extraction │    │  Discovery  │
                   └─────────────┘    └─────────────┘
```

**Process Flow:**
1. Detect log format (syslog, Apache, JSON, CSV, generic)
2. Parse each line to extract timestamp, severity, service name
3. Run Drain algorithm to extract template patterns
4. Store raw log + template reference in database
5. Create/update template entries

#### `query_service.py` - Log Queries
Handles filtered log retrieval with:
- Time range filtering
- Service name filtering
- Severity level filtering
- Template-based filtering
- Pagination with efficient counting

#### `embedding_service.py` - Vector Embeddings

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Template   │───▶│   Ollama    │───▶│   Vector    │───▶│   FAISS     │
│    Text     │    │  Embedding  │    │   (384-dim) │    │   Index     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**Purpose:**
- Converts log templates to 384-dimensional vectors
- Uses `nomic-embed-text` model via Ollama
- Enables semantic similarity search

#### `semantic_search_service.py` - AI Search
1. Convert user query to embedding vector
2. Search FAISS index for similar templates
3. Return matching logs with similarity scores

#### `chat_service.py` - LLM Integration
Provides conversational interface:
1. Retrieve relevant templates via semantic search
2. Build context with log statistics
3. Send to Qwen 2.5 LLM for analysis
4. Return natural language response

---

## The Drain Algorithm - Log Template Extraction

### What is Drain?

Drain is an online log parsing algorithm that extracts log templates (patterns) from raw log messages in real-time. It identifies variable parts (like IP addresses, timestamps, user IDs) and replaces them with `<*>` placeholders.

### How It Works

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Drain Algorithm Process                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: "Connection from 192.168.1.100 on port 22 accepted"                │
│                                    │                                        │
│                                    ▼                                        │
│  Step 1: Preprocessing ─────────────────────────────────────────────────── │
│          • Tokenize: ["Connection", "from", "192.168.1.100", "on", ...]    │
│          • Get length: 7 tokens                                             │
│                                    │                                        │
│                                    ▼                                        │
│  Step 2: Search Parse Tree ─────────────────────────────────────────────── │
│          • Navigate by: Length → First Token → Last Token                   │
│          • Find matching log group or create new                            │
│                                    │                                        │
│                                    ▼                                        │
│  Step 3: Template Matching ─────────────────────────────────────────────── │
│          • Compare with existing templates in group                         │
│          • Calculate similarity score                                       │
│          • If match found: update template                                  │
│          • If no match: create new template                                 │
│                                    │                                        │
│                                    ▼                                        │
│  Output Template: "Connection from <*> on port <*> accepted"               │
│  Variables: ["192.168.1.100", "22"]                                        │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Implementation (`apps/api/app/parsers/drain_parser.py`)

```python
class DrainParser:
    def __init__(self):
        self.drain = Drain(
            depth=4,           # Parse tree depth
            sim_th=0.4,        # Similarity threshold
            max_children=100,  # Max children per node
        )
    
    def parse(self, log_line: str) -> ParsedLog:
        result = self.drain.add_log_message(log_line)
        return ParsedLog(
            template=result.get_template(),
            template_hash=hash(result.get_template()),
            parameters=result.get_params()
        )
```

### Example Transformations

| Raw Log | Template | Variables |
|---------|----------|-----------|
| `User admin logged in from 10.0.0.1` | `User <*> logged in from <*>` | ["admin", "10.0.0.1"] |
| `Error: File /var/log/app.log not found` | `Error: File <*> not found` | ["/var/log/app.log"] |
| `Request took 245ms` | `Request took <*>ms` | ["245"] |

---

## Vector Embeddings & Semantic Search

### What Are Embeddings?

Embeddings are numerical vector representations of text that capture semantic meaning. Similar concepts have vectors that are close together in the embedding space.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Text to Vector Transformation                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Text: "Connection timeout error"                                          │
│                    │                                                        │
│                    ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │               Nomic Embed Text Model (via Ollama)                   │  │
│  │                                                                      │  │
│  │   • Neural network with attention mechanism                         │  │
│  │   • Trained on massive text corpus                                  │  │
│  │   • Outputs 384-dimensional vector                                   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                    │                                                        │
│                    ▼                                                        │
│  Vector: [0.023, -0.145, 0.089, 0.234, ..., -0.067]  (384 dimensions)     │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### FAISS Vector Index

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search of dense vectors.

```python
# apps/api/app/vector/faiss_index.py

class FAISSIndex:
    def __init__(self, dimension=384):
        self.index = faiss.IndexFlatIP(dimension)  # Inner product similarity
    
    def add(self, vectors: np.ndarray, ids: List[int]):
        """Add vectors to the index"""
        faiss.normalize_L2(vectors)  # Normalize for cosine similarity
        self.index.add(vectors)
    
    def search(self, query_vector: np.ndarray, k: int = 10):
        """Find k most similar vectors"""
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, k)
        return indices[0], scores[0]
```

### Semantic Search Flow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Semantic Search Process                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Query: "network connection problems"                                  │
│                    │                                                        │
│                    ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  1. Generate Query Embedding                                         │  │
│  │     query_vector = ollama.embed("network connection problems")       │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                    │                                                        │
│                    ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  2. Search FAISS Index                                               │  │
│  │     similar_templates = faiss_index.search(query_vector, k=20)       │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                    │                                                        │
│                    ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  3. Retrieve Matching Logs                                           │  │
│  │     logs = db.query(template_hash IN similar_template_hashes)        │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                    │                                                        │
│                    ▼                                                        │
│  Results:                                                                   │
│    • "Connection to <*> timed out" (score: 0.89)                           │
│    • "Network interface <*> down" (score: 0.85)                            │
│    • "Socket connection refused" (score: 0.82)                             │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Database Schema

### SQLite Database (`data/logmind.sqlite`)

#### `logs_stream` Table
Stores all ingested log events.

```sql
CREATE TABLE logs_stream (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    service_name TEXT NOT NULL,
    environment TEXT DEFAULT 'prod',
    timestamp_utc TEXT NOT NULL,
    ingest_timestamp_utc TEXT,
    severity INTEGER DEFAULT 2,
    host TEXT,
    template_hash INTEGER,          -- Links to log_templates
    parameters_json TEXT,            -- Extracted variables as JSON
    trace_id TEXT,
    span_id TEXT,
    attributes_json TEXT,            -- Additional metadata
    body_raw TEXT NOT NULL           -- Original log message
);

-- Indexes for fast queries
CREATE INDEX idx_logs_time_desc ON logs_stream (tenant_id, timestamp_utc DESC);
CREATE INDEX idx_logs_main ON logs_stream (tenant_id, service_name, timestamp_utc, severity);
CREATE INDEX idx_logs_template ON logs_stream (tenant_id, service_name, template_hash);
```

#### `log_templates` Table
Stores discovered log patterns.

```sql
CREATE TABLE log_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    service_name TEXT NOT NULL,
    template_hash TEXT NOT NULL,     -- Stored as string for 64-bit precision
    template_text TEXT NOT NULL,
    first_seen TEXT,
    last_seen TEXT,
    occurrence_count INTEGER DEFAULT 1,
    embedding_state TEXT DEFAULT 'pending',  -- pending/processing/done/error
    UNIQUE(tenant_id, service_name, template_hash)
);
```

#### `ingested_files` Table
Tracks processed files to avoid duplicates.

```sql
CREATE TABLE ingested_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    events_count INTEGER,
    UNIQUE(file_path, file_hash)
);
```

### Entity Relationship

```
┌─────────────────┐         ┌─────────────────┐
│   logs_stream   │         │  log_templates  │
├─────────────────┤         ├─────────────────┤
│ id              │         │ id              │
│ tenant_id       │────┐    │ tenant_id       │
│ service_name    │    │    │ service_name    │
│ template_hash   │────┼───▶│ template_hash   │
│ body_raw        │    │    │ template_text   │
│ timestamp_utc   │    │    │ embedding_state │
│ severity        │    │    │ occurrence_count│
│ ...             │    │    └─────────────────┘
└─────────────────┘    │
                       │    ┌─────────────────┐
                       │    │  ingested_files │
                       │    ├─────────────────┤
                       │    │ file_path       │
                       │    │ file_hash       │
                       └───▶│ events_count    │
                            └─────────────────┘
```

---

## Frontend Architecture (Next.js)

### Component Hierarchy

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              App Layout                                     │
│  ┌──────────────────┐  ┌─────────────────────────────────────────────────┐ │
│  │     Sidebar      │  │                   Main Content                   │ │
│  │  ┌────────────┐  │  │  ┌─────────────────────────────────────────────┐ │ │
│  │  │ Navigation │  │  │  │              Header (Filters)                │ │ │
│  │  │   Tabs     │  │  │  │  • Service Selector                        │ │ │
│  │  └────────────┘  │  │  │  • Time Range                              │ │ │
│  │  ┌────────────┐  │  │  │  • Upload/Train Buttons                    │ │ │
│  │  │ Statistics │  │  │  └─────────────────────────────────────────────┘ │ │
│  │  │   Panel    │  │  │  ┌─────────────────────────────────────────────┐ │ │
│  │  └────────────┘  │  │  │            Active View                      │ │ │
│  │  ┌────────────┐  │  │  │  • LogExplorer (browse logs)               │ │ │
│  │  │ AI Status  │  │  │  │  • TemplatesView (patterns)                │ │ │
│  │  │   Panel    │  │  │  │  • StatsView (analytics)                   │ │ │
│  │  └────────────┘  │  │  │  • SemanticSearchView (AI search)          │ │ │
│  └──────────────────┘  │  │  • AnomalyView (detection)                 │ │ │
│                        │  │  • AIModelView (training)                   │ │ │
│                        │  └─────────────────────────────────────────────┘ │ │
│                        └─────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

#### `src/app/page.tsx` - Main Application Page
- Manages global state (selected service, time range)
- Handles file upload and ingestion
- Controls tab navigation
- Displays embedding training status

#### `src/components/LogExplorer.tsx` - Log Browser
- Displays paginated log entries
- Supports search and filtering
- Expandable rows with detailed view
- Toggle between raw logs and templates

#### `src/components/ServiceSelector.tsx` - Service Dropdown
- Searchable dropdown for service selection
- Shows all available services from ingested logs

#### `src/components/SemanticSearchView.tsx` - AI Search
- Natural language search input
- Displays results with similarity scores
- Links to related logs

#### `src/components/AnomalyView.tsx` - Anomaly Detection
- Runs anomaly detection analysis
- Displays unusual patterns with severity scores

### API Client (`src/lib/api.ts`)

```typescript
// Centralized API client with timeout handling

const API_BASE = '/api';  // Proxied to localhost:8000

export async function queryLogs(params: LogQueryParams): Promise<LogQueryResponse> {
  return fetchJson(`${API_BASE}/logs?${buildSearchParams(params)}`, {
    timeout: 60000  // 60 second timeout for large datasets
  });
}

export async function semanticSearch(params: SearchParams): Promise<SearchResponse> {
  return fetchJson(`${API_BASE}/semantic_search?${buildSearchParams(params)}`);
}

export async function getQuickStats(): Promise<Stats> {
  return fetchJson(`${API_BASE}/logs/stats/quick`);
}
```

---

## Cross-Domain Few-Shot Learning

LogMind AI uses **cross-domain few-shot learning** to provide intelligent log analysis. This means the AI can learn patterns from one type of log (e.g., Apache web server) and apply that knowledge to analyze completely different log types (e.g., SSH authentication logs).

### What is Cross-Domain Few-Shot Learning?

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    CROSS-DOMAIN FEW-SHOT LEARNING                          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Traditional AI:           Cross-Domain Few-Shot:                         │
│   ┌─────────────┐           ┌─────────────────────────────────────────┐   │
│   │ Train on    │           │ Learn patterns from MULTIPLE domains:   │   │
│   │ Apache logs │──► Only   │                                         │   │
│   │             │   works   │   • Web Server patterns                 │   │
│   └─────────────┘   for     │   • Authentication patterns             │   │
│         ↓           Apache  │   • Network patterns                    │   │
│   ┌─────────────┐           │   • Database patterns                   │   │
│   │ Can't help  │           │   • Security patterns                   │   │
│   │ with SSH    │           │                                         │   │
│   │ or Network  │           │   These patterns TRANSFER to new logs!  │   │
│   └─────────────┘           └─────────────────────────────────────────┘   │
│                                         ↓                                  │
│                             ┌─────────────────────────────────────────┐   │
│                             │ AI can analyze ANY log type by applying │   │
│                             │ learned patterns from similar domains    │   │
│                             └─────────────────────────────────────────┘   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Supported Domains

LogMind recognizes and transfers knowledge between these log domains:

| Domain | Description | Example Services |
|--------|-------------|------------------|
| **web_server** | Web server logs | Apache, Nginx, IIS |
| **authentication** | Auth & access logs | SSH, Kerberos, LDAP |
| **system** | OS-level logs | Linux kernel, systemd, cron |
| **network** | Connectivity logs | Firewall, TCP, DNS |
| **database** | Database logs | MySQL, PostgreSQL, MongoDB |
| **application** | App-level logs | Java, Python, Node.js apps |
| **security** | Threat/attack logs | WAF, IDS, audit logs |
| **distributed** | Cluster logs | Hadoop, Spark, Kubernetes |

### How It Works

#### 1. Domain Detection

When you ask a question, the AI automatically detects which domains are relevant:

```python
# Example: "Why are there authentication failures in SSH?"

detected_domains = [
    ("authentication", 0.24),  # Primary domain
    ("security", 0.15),        # Related domain
]
```

#### 2. Example Selection

The system selects relevant training examples from multiple domains:

```python
# For an auth failure question, it might select:
examples = [
    {
        "source_domain": "authentication",
        "question": "Why are there so many authentication failures?",
        "transferable_patterns": [
            "Repeated failures from same source indicate automated attacks",
            "Time-based patterns reveal attack windows",
            "Geographic clustering may indicate botnets"
        ]
    },
    {
        "source_domain": "security",  # Cross-domain example!
        "question": "Are we under attack?",
        "transferable_patterns": [
            "Attacks leave traces across multiple log types",
            "Failed attempts indicate reconnaissance phase"
        ]
    }
]
```

#### 3. Transfer Rules

Domain-specific rules help the AI connect patterns across log types:

```json
{
  "transfer_rules": [
    {
      "rule": "If SSH auth failures increase, check web login endpoints",
      "source": "authentication",
      "target": "web_server"
    },
    {
      "rule": "Network issues manifest as database connection pool exhaustion",
      "source": "network",
      "target": "database"
    },
    {
      "rule": "Attacks create correlated anomalies in multiple log types",
      "source": "security",
      "target": "all"
    }
  ]
}
```

#### 4. Enhanced Prompt Generation

The few-shot prompt combines examples + transfer rules + evidence:

```
=== EXAMPLE ANALYSES (Learn from these patterns) ===

**Example 1** (authentication domain):
Question: Why are there so many authentication failures?
Answer: [Detailed analysis with patterns...]

Key Patterns to Remember:
  • Repeated failures from same source indicate automated attacks
  • Time-based patterns reveal attack windows

=== CROSS-DOMAIN INSIGHTS ===
Apply these patterns when analyzing:
  • If SSH auth failures increase, check web login endpoints
  • Attacks create correlated anomalies in multiple log types

=== CURRENT ANALYSIS TASK ===
Service: LabSZ
Detected Domain(s): authentication, security

=== LOG EVIDENCE ===
[template:12345] Failed password for root from 106.5.5.195 port 50719 ssh2
...

=== USER QUESTION ===
Why are there authentication failures?
```

### Implementation Details

#### Files Structure

```
apps/api/app/services/
├── few_shot_service.py      # Main few-shot learning service
├── few_shot_examples.json   # Training examples database
└── chat_service.py          # Uses few-shot for chat responses
```

#### Code: Domain Detection

```python
# apps/api/app/services/few_shot_service.py

def detect_domains(self, question: str, service_name: str) -> List[Tuple[str, float]]:
    """Detect relevant domains from question and service."""
    text = f"{question} {service_name}".lower()
    
    domain_scores = {}
    for domain_name, domain_info in self.domains.items():
        keywords = domain_info.get("keywords", [])
        score = sum(1 for k in keywords if k in text)
        if score > 0:
            domain_scores[domain_name] = score / len(keywords)
    
    return sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
```

#### Code: Cross-Domain Example Selection

```python
def get_relevant_examples(self, detected_domains, question, max_examples=3):
    """Select examples using cross-domain relevance."""
    example_scores = []
    
    for example in self.examples:
        score = 0
        
        # Primary domain match: +3 points
        if example["source_domain"] == detected_domains[0][0]:
            score += 3
        
        # Applicable domain match: +2 points each
        for domain, confidence in detected_domains:
            if domain in example["applicable_domains"]:
                score += 2 * confidence
        
        example_scores.append((example, score))
    
    # Return top N by score
    return [ex for ex, _ in sorted(example_scores, key=lambda x: -x[1])[:max_examples]]
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/few-shot/stats` | GET | Get few-shot system statistics |
| `/chat/few-shot/domains?question=...` | GET | Detect domains for a question |
| `/chat/few-shot/examples?question=...` | GET | Get relevant training examples |

### Example API Usage

```bash
# Get system statistics
curl "http://localhost:8000/chat/few-shot/stats"
# Returns: {"total_domains": 8, "total_examples": 8, "total_transfer_rules": 5, ...}

# Detect domains for a question
curl "http://localhost:8000/chat/few-shot/domains?question=Why%20database%20connection%20timeout"
# Returns: {"detected_domains": [{"domain": "database", "confidence": 0.15}, {"domain": "network", "confidence": 0.26}]}

# Get relevant examples
curl "http://localhost:8000/chat/few-shot/examples?question=authentication%20failures&max_examples=2"
# Returns examples with transferable patterns
```

### Key Benefits

1. **Zero Training Required**: Works immediately with pre-defined patterns
2. **Domain Transfer**: Knowledge from one log type helps analyze others
3. **Pattern Recognition**: Learns to identify similar issues across systems
4. **Continuous Improvement**: Add new examples to improve coverage
5. **Explainable**: Shows which examples and rules influenced the analysis

---

## AI Model Training Pipeline

This section explains **how LogMind trains its AI model** on your log data. The training process converts your log templates into AI-searchable vectors.

### Training Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         AI TRAINING PIPELINE                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐                                                           │
│   │ LOG FILES   │ ──► Raw log files uploaded by user                       │
│   └──────┬──────┘                                                           │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ STEP 1: Log Ingestion & Parsing (Drain Algorithm)                   │  │
│   │                                                                      │  │
│   │   Raw Log: "Connection timeout from 192.168.1.1 at port 8080"       │  │
│   │              │                                                       │  │
│   │              ▼                                                       │  │
│   │   Template: "Connection timeout from <*> at port <*>"               │  │
│   │   Parameters: ["192.168.1.1", "8080"]                               │  │
│   │                                                                      │  │
│   │   • Similar logs are clustered into templates                       │  │
│   │   • 5.8M logs → ~47,000 unique templates                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ STEP 2: Template Storage in Database                                │  │
│   │                                                                      │  │
│   │   Database stores each template with:                               │  │
│   │   • template_hash (unique identifier)                               │  │
│   │   • template_text ("Connection timeout from <*>...")                │  │
│   │   • embedding_state: 'none' → 'queued' → 'ready'                    │  │
│   │   • first_seen_utc, last_seen_utc                                   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│          │                                                                  │
│          ▼     ◄──── User clicks "Train AI" button                         │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ STEP 3: Embedding Generation (The "Training")                       │  │
│   │                                                                      │  │
│   │   For each template with embedding_state = 'none':                  │  │
│   │                                                                      │  │
│   │   1. Send template text to Ollama nomic-embed-text model            │  │
│   │   2. Receive 384-dimensional vector representation                  │  │
│   │   3. Store vector in FAISS index                                    │  │
│   │   4. Mark template as embedding_state = 'ready'                     │  │
│   │                                                                      │  │
│   │   Processing: 100 templates per batch (configurable)                │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ STEP 4: Vector Index Ready for Search                               │  │
│   │                                                                      │  │
│   │   FAISS Index contains:                                             │  │
│   │   • 7,106 template vectors (currently embedded)                     │  │
│   │   • Each vector is 384 dimensions                                   │  │
│   │   • Enables semantic search in milliseconds                         │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Understanding "Training" in LogMind

**Important**: LogMind does NOT train the AI model itself. Instead:

1. **Pre-trained Model**: We use `nomic-embed-text`, which is already trained on billions of text samples
2. **Vector Generation**: We pass our log templates through this model to get vector representations
3. **Index Building**: These vectors are stored in FAISS for fast similarity search

This is similar to how you'd use a language model - you don't retrain it, you use its pre-learned knowledge to understand your data.

### Embedding State Machine

Each template goes through these states:

```
┌────────┐     ┌────────┐     ┌─────────────┐     ┌───────┐
│  none  │ ──► │ queued │ ──► │ processing  │ ──► │ ready │
└────────┘     └────────┘     └─────────────┘     └───────┘
     │                                                 ▲
     │              ┌────────┐                         │
     └─────────────►│ failed │ ◄──────────────────────┘
                    └────────┘     (on error, retry later)
```

### Training Code Walkthrough

**1. User initiates training via API:**

```python
# POST /api/chat/embed?max_templates=100

@router.post("/embed")
async def process_embeddings(max_templates: int = 100):
    """Process pending template embeddings."""
    service = get_embedding_service()
    count = await service.process_pending_templates(max_templates=max_templates)
    return {"processed": count}
```

**2. EmbeddingService fetches pending templates:**

```python
# apps/api/app/services/embedding_service.py

async def process_pending_templates(self, batch_size: int = 50, max_templates: int = 100):
    """Process templates that need embeddings."""
    
    # Get templates needing embeddings (state = 'none' or 'queued')
    templates = await templates_repo.get_templates_needing_embedding(max_templates)
    
    for template in templates:
        # Generate embedding via Ollama
        vector = await self.embed_text(template.template_text)
        
        if vector is None:
            await templates_repo.update_embedding_state(
                template_hash=template.template_hash,
                state="failed"
            )
            continue
        
        # Add vector to FAISS index
        faiss_id = await faiss_index.add_vector(
            template_hash=template.template_hash,
            vector=vector
        )
        
        # Store in database for persistence
        await vectors_repo.upsert_vector(
            template_hash=template.template_hash,
            faiss_id=faiss_id,
            vector=vector
        )
        
        # Mark as ready
        await templates_repo.update_embedding_state(
            template_hash=template.template_hash,
            state="ready",
            model="nomic-embed-text"
        )
    
    # Save FAISS index to disk
    await faiss_index.save()
    return processed_count
```

**3. Embedding generation via Ollama:**

```python
# apps/api/app/llm/ollama_client.py

async def embed(self, text: str) -> List[float]:
    """Generate embedding vector for text."""
    response = await self._request(
        "POST",
        "/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    return response["embedding"]  # Returns 384-dimensional vector
```

### Database Schema for Training

```sql
-- log_templates table tracks embedding status
CREATE TABLE log_templates (
    tenant_id TEXT NOT NULL,
    service_name TEXT NOT NULL,
    template_hash INTEGER NOT NULL,
    template_text TEXT NOT NULL,
    first_seen_utc TEXT,
    last_seen_utc TEXT,
    
    -- Training state columns
    embedding_state TEXT DEFAULT 'none',  -- 'none', 'queued', 'ready', 'failed'
    embedding_model TEXT,                  -- 'nomic-embed-text'
    embedding_updated_utc TEXT,
    
    PRIMARY KEY (tenant_id, service_name, template_hash)
);

-- template_vectors stores the actual vectors
CREATE TABLE template_vectors (
    tenant_id TEXT NOT NULL,
    service_name TEXT NOT NULL,
    template_hash INTEGER NOT NULL,
    faiss_id INTEGER NOT NULL,            -- Index position in FAISS
    vector BLOB NOT NULL,                 -- 384 floats stored as binary
    created_utc TEXT,
    
    PRIMARY KEY (tenant_id, service_name, template_hash)
);
```

### FAISS Index Details

```python
# apps/api/app/vector/faiss_index.py

class FAISSIndex:
    def __init__(self, dimension=384):
        # IndexFlatIP = Inner Product (dot product) for cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.dimension = dimension
    
    async def add_vector(self, tenant_id, service_name, template_hash, vector):
        """Add a single vector to the index."""
        # Normalize for cosine similarity
        faiss.normalize_L2(vector.reshape(1, -1))
        
        # Add to index
        faiss_id = self.index.ntotal  # Current index size = new ID
        self.index.add(vector.reshape(1, -1))
        
        # Store mapping: faiss_id -> (tenant_id, service_name, template_hash)
        self._id_mapping[faiss_id] = (tenant_id, service_name, template_hash)
        
        return faiss_id
    
    async def save(self):
        """Persist index to disk."""
        faiss.write_index(self.index, "data/faiss.index")
    
    async def load(self):
        """Load index from disk."""
        self.index = faiss.read_index("data/faiss.index")
```

### Monitoring Training Progress

The UI shows training progress via the `/api/chat/embed/stats` endpoint:

```python
async def get_embedding_stats(self) -> dict:
    """Get embedding generation statistics."""
    
    # Total templates
    total = SELECT COUNT(*) FROM log_templates
    
    # Already embedded
    embedded = SELECT COUNT(*) FROM log_templates 
               WHERE embedding_state = 'ready'
    
    # Waiting to be processed
    pending = SELECT COUNT(*) FROM log_templates 
              WHERE embedding_state IN ('none', 'queued')
    
    # Failed (will retry)
    failed = SELECT COUNT(*) FROM log_templates 
             WHERE embedding_state = 'failed'
    
    return {
        "total_templates": 47698,      # Example values
        "embedded_count": 7106,
        "pending_count": 40592,
        "failed_count": 0,
        "percentage": 14.9
    }
```

---

## AI Components Deep Dive

### 1. Embedding Generation

The system uses `nomic-embed-text` model for generating text embeddings:

```python
# apps/api/app/services/embedding_service.py

class EmbeddingService:
    def __init__(self):
        self._processing = False
        self._lock = asyncio.Lock()
    
    async def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Generate 384-dimensional embedding for text."""
        ollama = get_ollama_client()
        embedding_list = await ollama.embed(text)
        return list_to_vector(embedding_list)  # Convert to numpy array
```

### 2. LLM Chat Integration

```python
# apps/api/app/services/chat_service.py

class ChatService:
    def __init__(self):
        self.model = "qwen2.5:3b"
    
    async def chat(self, question: str, service: str, time_range: tuple):
        # 1. Get relevant context via semantic search
        relevant_templates = await self.semantic_search(question, k=10)
        
        # 2. Get statistics for context
        stats = await self.get_service_stats(service, time_range)
        
        # 3. Build prompt with context
        prompt = f"""You are an expert log analyst. Analyze these logs and answer the question.

        Service: {service}
        Time Range: {time_range}
        Total Logs: {stats.total_logs}
        Error Rate: {stats.error_rate}%
        
        Relevant Log Patterns:
        {self.format_templates(relevant_templates)}
        
        Question: {question}
        
        Provide a clear, technical analysis."""
        
        # 4. Get LLM response
        response = await self.ollama_generate(prompt)
        return response
```

### 3. Anomaly Detection

```python
# apps/api/app/services/anomaly_service.py

class AnomalyService:
    async def detect_anomalies(self, service: str, time_range: tuple):
        """Detect unusual log patterns"""
        
        # 1. Get template frequency distribution
        template_stats = await self.get_template_frequencies(service, time_range)
        
        # 2. Calculate baseline statistics
        mean_freq = np.mean(template_stats.frequencies)
        std_freq = np.std(template_stats.frequencies)
        
        # 3. Identify anomalies (Z-score > 2)
        anomalies = []
        for template in template_stats:
            z_score = (template.frequency - mean_freq) / std_freq
            if abs(z_score) > 2:
                anomalies.append({
                    "template": template.text,
                    "score": z_score,
                    "severity": "high" if z_score > 3 else "medium"
                })
        
        # 4. Use LLM for analysis
        if anomalies:
            analysis = await self.llm_analyze(anomalies)
            return {"anomalies": anomalies, "analysis": analysis}
```

---

## Data Flow Diagrams

### Log Ingestion Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Log Ingestion Flow                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────┐                                                               │
│   │Log File │                                                               │
│   │(.log)   │                                                               │
│   └────┬────┘                                                               │
│        │                                                                     │
│        ▼                                                                     │
│   ┌────────────────┐    ┌────────────────┐    ┌────────────────────┐       │
│   │ Format         │───▶│ Parse Line     │───▶│ Drain Algorithm    │       │
│   │ Detection      │    │ • Timestamp    │    │ • Extract Template │       │
│   │ • Syslog       │    │ • Severity     │    │ • Calculate Hash   │       │
│   │ • Apache       │    │ • Service      │    │ • Get Parameters   │       │
│   │ • JSON         │    │ • Message      │    │                    │       │
│   └────────────────┘    └────────────────┘    └─────────┬──────────┘       │
│                                                          │                  │
│                              ┌───────────────────────────┘                  │
│                              │                                              │
│                              ▼                                              │
│   ┌────────────────────────────────────────────────────────────────┐       │
│   │                    SQLite Database                              │       │
│   │  ┌──────────────────┐    ┌──────────────────────┐             │       │
│   │  │ logs_stream      │    │ log_templates         │             │       │
│   │  │ • body_raw       │    │ • template_text       │             │       │
│   │  │ • template_hash ─┼───▶│ • template_hash       │             │       │
│   │  │ • parameters     │    │ • embedding_state     │             │       │
│   │  └──────────────────┘    └──────────────────────┘             │       │
│   └────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### AI Training Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AI Training Flow                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────────┐                                                    │
│   │ log_templates      │                                                    │
│   │ (pending state)    │                                                    │
│   └─────────┬──────────┘                                                    │
│             │ Batch of 100                                                  │
│             ▼                                                               │
│   ┌────────────────────┐    ┌────────────────────┐                        │
│   │ Template Text      │───▶│ Ollama API         │                        │
│   │ "User <*> login"   │    │ /api/embeddings    │                        │
│   └────────────────────┘    │                    │                        │
│                             │ nomic-embed-text   │                        │
│                             └─────────┬──────────┘                        │
│                                       │                                     │
│                                       ▼                                     │
│                             ┌────────────────────┐                        │
│                             │ 384-dim Vector     │                        │
│                             │ [0.12, -0.05, ...] │                        │
│                             └─────────┬──────────┘                        │
│                                       │                                     │
│              ┌────────────────────────┼────────────────────┐               │
│              ▼                        ▼                    ▼               │
│   ┌────────────────────┐   ┌────────────────────┐  ┌──────────────┐      │
│   │ FAISS Index        │   │ Template Mapping   │  │ Update State │      │
│   │ • Add vector       │   │ • Vector ID → Hash │  │ • "done"     │      │
│   │ • Persist to disk  │   │ • For reverse      │  │              │      │
│   │                    │   │   lookup           │  │              │      │
│   └────────────────────┘   └────────────────────┘  └──────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Semantic Search Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Semantic Search Flow                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   User: "show me authentication failures"                                   │
│                    │                                                         │
│                    ▼                                                         │
│   ┌────────────────────┐                                                    │
│   │ Embed Query        │                                                    │
│   │ → [0.23, -0.11...]│                                                    │
│   └─────────┬──────────┘                                                    │
│             │                                                                │
│             ▼                                                                │
│   ┌────────────────────┐                                                    │
│   │ FAISS Search       │                                                    │
│   │ • Cosine similarity│                                                    │
│   │ • Top 20 results   │                                                    │
│   └─────────┬──────────┘                                                    │
│             │                                                                │
│             ▼                                                                │
│   ┌────────────────────────────────────────────────────────────────┐       │
│   │ Matched Templates (by similarity score)                        │       │
│   │                                                                 │       │
│   │ 1. "Authentication failed for user <*>" (0.92)                │       │
│   │ 2. "Login failed from IP <*>" (0.87)                          │       │
│   │ 3. "Invalid password for <*>" (0.85)                          │       │
│   └─────────────────────────┬──────────────────────────────────────┘       │
│                             │                                               │
│                             ▼                                               │
│   ┌────────────────────────────────────────────────────────────────┐       │
│   │ Retrieve Matching Logs                                          │       │
│   │ SELECT * FROM logs_stream                                       │       │
│   │ WHERE template_hash IN (matched_hashes)                         │       │
│   │ ORDER BY timestamp DESC                                         │       │
│   └────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Performance Optimizations

### Database Optimizations

1. **SQLite WAL Mode**: Enables concurrent reads during writes
2. **Memory-Mapped I/O**: 256MB mmap for faster disk access
3. **Large Cache**: 20,000 pages (~80MB) for query caching
4. **Optimized Indexes**: Covering indexes for common query patterns

```python
# apps/api/app/storage/db.py
PRAGMAS = {
    "journal_mode": "WAL",
    "synchronous": "NORMAL",
    "cache_size": 20000,
    "mmap_size": 268435456,  # 256MB
    "temp_store": "MEMORY",
}
```

### Query Optimizations

1. **Efficient Counting**: Avoids COUNT(*) on large tables
2. **Has-More Detection**: Checks for next page existence instead of total
3. **Index-Optimized Queries**: Uses covering indexes

```python
# Instead of slow COUNT(*)
if len(logs) < limit:
    total = offset + len(logs)
else:
    # Check if there's one more row (O(1) vs O(n))
    has_more = await db.execute("SELECT 1 ... LIMIT 1 OFFSET ?", offset+limit)
```

---

## API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/logs` | GET | Query logs with filters |
| `/logs/stats/quick` | GET | Fast statistics |
| `/logs/services/list` | GET | List all services |
| `/templates/top` | GET | Top templates by count |
| `/semantic_search` | GET | AI-powered search |
| `/chat` | POST | Chat with LLM |
| `/ingest/file` | POST | Upload and ingest file |
| `/anomaly/detect` | GET | Run anomaly detection |

### Example Requests

```bash
# Query logs
curl "http://localhost:8000/logs?from=2024-01-01&to=2024-12-31&limit=50"

# Semantic search
curl "http://localhost:8000/semantic_search?q=authentication%20error&service_name=sshd"

# Chat
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main error patterns?", "service_name": "apache"}'
```

---

## Running the Application

### Prerequisites

1. Python 3.11+ with pip
2. Node.js 18+ with npm
3. Ollama with required models:
   ```bash
   ollama pull qwen2.5:3b
   ollama pull nomic-embed-text
   ```

### Start the Application

```bash
cd /home/bug/Desktop/Log_Analyzer
./start.sh
```

### Access Points

- **Dashboard**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## Current Statistics

| Metric | Value |
|--------|-------|
| Total Logs | ~5,831,375 |
| Templates | ~47,698 |
| Services | 374 |
| Trained Embeddings | ~7,106 |
| Database Size | ~2.2 GB |

---

## Conclusion

LogMind AI demonstrates a complete implementation of a modern log analytics platform combining:

1. **Traditional Log Parsing** - Using the proven Drain algorithm
2. **Machine Learning** - Vector embeddings for semantic understanding
3. **Large Language Models** - Natural language interaction and analysis
4. **Efficient Storage** - Optimized SQLite for millions of records
5. **Modern UI** - Responsive React-based dashboard

The system provides a foundation for intelligent log analysis that can scale to handle enterprise log volumes while maintaining sub-second query response times.

---

*Document Version: 1.0*
*Last Updated: December 23, 2025*
*Project: LogMind AI - Log Observability Platform*
