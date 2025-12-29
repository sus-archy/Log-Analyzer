# LogMind AI - Complete Code Documentation

## Table of Contents

### Part 1: Core Infrastructure
1. [Application Entry Point (main.py)](#1-application-entry-point-mainpy)
2. [Configuration System (config.py)](#2-configuration-system-configpy)
3. [Logging System (logging.py)](#3-logging-system-loggingpy)
4. [Rate Limiting (rate_limit.py)](#4-rate-limiting-rate_limitpy)
5. [Caching System (cache.py)](#5-caching-system-cachepy)
6. [Security & Authentication (security.py)](#6-security--authentication-securitypy)

### Part 2: Storage & Database Layer
7. [Database Connection Pool (db.py)](#7-database-connection-pool-dbpy)
8. [Schema Migrations (migrations.py)](#8-schema-migrations-migrationspy)
9. [Logs Repository (logs_repo.py)](#9-logs-repository-logs_repopy)
10. [Templates Repository (templates_repo.py)](#10-templates-repository-templates_repopy)

### Part 3: Parsers & Log Processing
11. [Drain Template Miner (drain_miner.py)](#11-drain-template-miner-drain_minerpy)
12. [Text Parser (text_parser.py)](#12-text-parser-text_parserpy)
13. [JSONL Parser (jsonl_parser.py)](#13-jsonl-parser-jsonl_parserpy)
14. [Normalizer (normalize.py)](#14-normalizer-normalizepy)

### Part 4: Machine Learning Models
15. [Anomaly Detector (anomaly_detector.py)](#15-anomaly-detector-anomaly_detectorpy)
16. [Log Classifier (log_classifier.py)](#16-log-classifier-log_classifierpy)
17. [Security Threat Detector (security_threat_detector.py)](#17-security-threat-detector-security_threat_detectorpy)
18. [Training Pipeline (training_pipeline.py)](#18-training-pipeline-training_pipelinepy)

### Part 5: API Routes & Services
19. [ML Routes (ml.py)](#19-ml-routes-mlpy)
20. [Ingest Service (ingest_service.py)](#20-ingest-service-ingest_servicepy)
21. [Chat Service (chat_service.py)](#21-chat-service-chat_servicepy)
22. [Ollama Client (ollama_client.py)](#22-ollama-client-ollama_clientpy)

### Part 6: Frontend Components
23. [API Client (api.ts)](#23-api-client-apits)
24. [AIChat Component (AIChat.tsx)](#24-aichat-component-aichattsx)
25. [LogExplorer Component (LogExplorer.tsx)](#25-logexplorer-component-logexplorertsx)

---

# Part 1: Core Infrastructure

## 1. Application Entry Point (main.py)

**File:** `apps/api/app/main.py`

### Purpose
This is the main entry point for the FastAPI application. It initializes all services, configures middleware, and sets up the application lifecycle.

### Code Structure

```python
"""
LogMind AI - FastAPI Application
Local log observability platform with template mining and semantic search.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
```

### Key Components

#### 1.1 Lifespan Handler
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # STARTUP PHASE
    logger.info("Starting LogMind AI...")
    
    # 1. Initialize database connection pool
    await init_db()
    
    # 2. Initialize FAISS vector index
    await init_faiss_index()
    
    # 3. Load/rebuild embeddings for semantic search
    embedding_service = get_embedding_service()
    await embedding_service.ensure_index_loaded()
    
    # 4. Initialize ML models (non-blocking)
    try:
        from .ml import initialize_models
        ml_status = await initialize_models()
    except Exception as e:
        logger.warning(f"ML model initialization failed: {e}")
    
    yield  # Application runs here
    
    # SHUTDOWN PHASE
    await get_cache().clear()
    await close_db()
    await close_ollama_client()
```

**Explanation:**
- Uses Python's `asynccontextmanager` for async resource management
- **Startup**: Initializes DB, FAISS index, embeddings, and ML models
- **Yield**: Application serves requests while context is active
- **Shutdown**: Cleans up cache, closes DB connections, closes Ollama client

#### 1.2 FastAPI App Creation
```python
app = FastAPI(
    title="LogMind AI",
    description="Local log observability platform",
    version="1.0.0",
    lifespan=lifespan,
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

**Explanation:**
- Creates FastAPI app with metadata for OpenAPI docs
- Attaches rate limiter to `app.state` for access in routes
- Registers custom exception handler for rate limit errors

#### 1.3 CORS Middleware Configuration
```python
cors_origins = [origin.strip() for origin in settings.cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,          # Allowed origins from config
    allow_credentials=True,               # Allow cookies
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    max_age=3600,                         # Cache preflight for 1 hour
)
```

**Explanation:**
- Parses comma-separated CORS origins from settings
- Configures Cross-Origin Resource Sharing for frontend access
- Exposes custom headers for request tracking and rate limiting

#### 1.4 Router Registration
```python
# Authentication
app.include_router(auth_router)

# Core endpoints
app.include_router(health_router)
app.include_router(metrics_router)

# Data management
app.include_router(ingest_router)
app.include_router(logs_router)
app.include_router(templates_router)

# AI & Search
app.include_router(semantic_router)
app.include_router(chat_router)
app.include_router(ml_router)
```

**Explanation:**
- Routers are organized by category
- Each router handles a specific domain (auth, logs, ML, etc.)
- Routes are automatically prefixed based on router configuration

---

## 2. Configuration System (config.py)

**File:** `apps/api/app/core/config.py`

### Purpose
Centralized configuration management using Pydantic Settings. Loads environment variables with type safety and validation.

### Code Structure

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",           # Load from .env file
        env_file_encoding="utf-8",
        case_sensitive=False,       # ENV vars are case-insensitive
        extra="ignore"              # Ignore unknown env vars
    )
```

### Configuration Categories

#### 2.1 Database Configuration
```python
# Database paths - computed from project root
db_path: str = str(Path(__file__).parent.parent.parent.parent.parent / "data" / "logmind.sqlite")
faiss_index_path: str = str(Path(__file__).parent.parent.parent.parent.parent / "data" / "faiss.index")
models_dir: str = str(Path(__file__).parent.parent.parent.parent.parent / "data" / "models")

# Connection pool settings
db_pool_size: int = 5           # Max concurrent connections
db_pool_timeout: float = 30.0   # Timeout waiting for connection
```

**Explanation:**
- Uses `Path(__file__)` to compute paths relative to project root
- This ensures paths work regardless of working directory
- Pool settings control database connection management

#### 2.2 Ollama Configuration
```python
ollama_base_url: str = "http://127.0.0.1:11434"
ollama_chat_model: str = "qwen2.5:3b"        # Model for chat
ollama_embed_model: str = "nomic-embed-text"  # Model for embeddings
ollama_api_key: Optional[str] = None          # Optional API key
```

**Explanation:**
- Configures connection to local Ollama LLM server
- Separate models for chat (reasoning) and embeddings (vectors)
- API key is optional for local deployment

#### 2.3 Security Configuration
```python
auth_enabled: bool = False  # Disabled by default for dev
secret_key: str = "CHANGE-THIS-SECRET-KEY-IN-PRODUCTION"
algorithm: str = "HS256"
access_token_expire_minutes: int = 1440  # 24 hours
admin_username: str = "admin"
admin_password_hash: str = "admin"
```

**Explanation:**
- Auth is disabled by default for easy development
- JWT settings for token generation
- Default admin credentials (MUST change in production)

#### 2.4 Rate Limiting Configuration
```python
rate_limit_enabled: bool = True
rate_limit_default: str = "100/minute"
rate_limit_chat: str = "10/minute"     # Stricter for LLM calls
rate_limit_embed: str = "60/minute"    # Allow batch embedding
rate_limit_ingest: str = "20/minute"
rate_limit_search: str = "30/minute"
```

**Explanation:**
- Different rate limits for different endpoints
- Chat has strict limits (LLM is expensive)
- Ingest has moderate limits (write operations)

#### 2.5 Property Methods
```python
@property
def db_path_resolved(self) -> Path:
    """Get absolute path to database."""
    return Path(self.db_path).resolve()

@property
def faiss_index_path_resolved(self) -> Path:
    """Get absolute path to FAISS index."""
    return Path(self.faiss_index_path).resolve()
```

**Explanation:**
- Properties provide resolved absolute paths
- `resolve()` handles symlinks and normalizes paths

#### 2.6 Cached Settings Instance
```python
@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

# Convenience alias
settings = get_settings()
```

**Explanation:**
- `@lru_cache()` ensures settings are loaded only once
- Single instance is reused across the application
- `settings` alias provides easy import

---

## 3. Logging System (logging.py)

**File:** `apps/api/app/core/logging.py`

### Purpose
Configures structured logging with custom formatters for the application.

### Implementation

```python
import logging
import sys
from typing import Optional

def setup_logging(level: str = "INFO") -> None:
    """Configure application logging."""
    
    # Create formatter with timestamp, level, and module
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(console_handler)
    
    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
```

**Explanation:**
- Creates consistent log format across the application
- Silences verbose third-party libraries
- `get_logger()` provides module-specific loggers

---

## 4. Rate Limiting (rate_limit.py)

**File:** `apps/api/app/core/rate_limit.py`

### Purpose
Implements per-endpoint rate limiting to prevent abuse and ensure fair resource usage.

### Implementation

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from ..core.config import settings

def get_rate_limit_key(request) -> str:
    """Get rate limit key from request."""
    # Use IP address as the key
    return get_remote_address(request)

# Create limiter instance
limiter = Limiter(
    key_func=get_rate_limit_key,
    enabled=settings.rate_limit_enabled,
    default_limits=[settings.rate_limit_default]
)
```

### Usage in Routes

```python
from ..core.rate_limit import limiter

@router.post("/chat")
@limiter.limit(settings.rate_limit_chat)  # "10/minute"
async def chat(request: Request, ...):
    ...
```

**Explanation:**
- Uses `slowapi` library for rate limiting
- Key function extracts client IP address
- Decorators apply per-endpoint limits
- Headers expose remaining quota to clients

---

## 5. Caching System (cache.py)

**File:** `apps/api/app/core/cache.py`

### Purpose
In-memory TTL (Time-To-Live) cache for frequently accessed data.

### Implementation

```python
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import asyncio

class TTLCache:
    """Simple in-memory cache with TTL expiration."""
    
    def __init__(self, default_ttl: int = 300):  # 5 minutes default
        self.default_ttl = default_ttl
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        async with self._lock:
            if key in self._cache:
                value, expires_at = self._cache[key]
                if datetime.now() < expires_at:
                    return value
                # Expired - remove it
                del self._cache[key]
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        async with self._lock:
            expires_at = datetime.now() + timedelta(seconds=ttl or self.default_ttl)
            self._cache[key] = (value, expires_at)
    
    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        async with self._lock:
            self._cache.pop(key, None)
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()

# Singleton instance
_cache: Optional[TTLCache] = None

def get_cache() -> TTLCache:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        _cache = TTLCache()
    return _cache
```

**Explanation:**
- Thread-safe async cache with `asyncio.Lock`
- Each entry stores value + expiration timestamp
- Lazy cleanup on `get()` - expired entries removed when accessed
- Singleton pattern ensures single cache instance

---

## 6. Security & Authentication (security.py)

**File:** `apps/api/app/core/security.py`

### Purpose
JWT-based authentication and password hashing.

### Implementation

```python
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from .config import settings

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[dict]:
    """Get current user from JWT token."""
    if not settings.auth_enabled:
        return {"username": "anonymous", "is_admin": True}
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"username": username, "is_admin": payload.get("is_admin", False)}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

**Explanation:**
- Uses `bcrypt` for secure password hashing
- JWT tokens with configurable expiration
- When auth disabled, returns anonymous admin user
- `Depends(oauth2_scheme)` extracts token from Authorization header








---

*End of Part 1. Continue with Part 2 for Storage & Database Layer.*

---








# Part 2: Storage & Database Layer

## 7. Database Connection Pool (db.py)

**File:** `apps/api/app/storage/db.py`

### Purpose
Provides an async connection pool for SQLite using `aiosqlite`. Manages concurrent database access with optimized settings.

### Class: ConnectionPool

```python
class ConnectionPool:
    """
    Simple async connection pool for aiosqlite.
    Manages a pool of database connections for concurrent access.
    """
    
    def __init__(self, db_path: Path, pool_size: int = 5, timeout: float = 30.0):
        self.db_path = db_path
        self.pool_size = pool_size
        self.timeout = timeout
        self._pool: deque[aiosqlite.Connection] = deque()  # Available connections
        self._in_use: set[aiosqlite.Connection] = set()    # Checked out connections
        self._lock = asyncio.Lock()                         # Thread safety
        self._initialized = False
        self._closed = False
```

**Explanation:**
- `deque` provides O(1) append/pop operations for connection queue
- `_in_use` set tracks which connections are currently being used
- `asyncio.Lock` ensures thread-safe pool operations

### 7.1 Connection Creation with SQLite Optimizations

```python
async def _create_connection(self) -> aiosqlite.Connection:
    """Create a new optimized database connection."""
    conn = await aiosqlite.connect(str(self.db_path), timeout=60.0)
    conn.row_factory = aiosqlite.Row  # Return dict-like rows
    
    # Performance optimizations via PRAGMA statements
    await conn.execute("PRAGMA busy_timeout=60000")      # Wait 60s for locks
    await conn.execute("PRAGMA journal_mode=WAL")        # Write-Ahead Logging
    await conn.execute("PRAGMA synchronous=NORMAL")      # Balance durability/speed
    await conn.execute("PRAGMA cache_size=-200000")      # 200MB cache
    await conn.execute("PRAGMA temp_store=MEMORY")       # Temp tables in RAM
    await conn.execute("PRAGMA mmap_size=268435456")     # 256MB memory-mapped I/O
    await conn.execute("PRAGMA read_uncommitted=1")      # Allow dirty reads
    await conn.execute("PRAGMA wal_autocheckpoint=1000") # Checkpoint every 1000 pages
    
    return conn
```

**PRAGMA Explanations:**

| PRAGMA | Value | Purpose |
|--------|-------|---------|
| `busy_timeout` | 60000 | Wait 60 seconds when database is locked |
| `journal_mode` | WAL | Enables concurrent reads during writes |
| `synchronous` | NORMAL | Sync less frequently (faster, slightly less safe) |
| `cache_size` | -200000 | 200MB page cache (negative = KB) |
| `temp_store` | MEMORY | Store temp tables in RAM |
| `mmap_size` | 256MB | Memory-map the database file |
| `read_uncommitted` | 1 | Allow reading uncommitted data |
| `wal_autocheckpoint` | 1000 | Auto-checkpoint after 1000 pages |

### 7.2 Connection Acquisition

```python
async def acquire(self) -> aiosqlite.Connection:
    """Acquire a connection from the pool."""
    if self._closed:
        raise RuntimeError("Connection pool is closed")
    
    start_time = asyncio.get_event_loop().time()
    
    while True:
        async with self._lock:
            # Try to get from pool
            if self._pool:
                conn = self._pool.popleft()
                # Verify connection is alive
                try:
                    await conn.execute("SELECT 1")
                    self._in_use.add(conn)
                    return conn
                except Exception:
                    # Connection dead, close and try again
                    try:
                        await conn.close()
                    except Exception:
                        pass
                    continue
            
            # Create new if pool not full
            total_connections = len(self._pool) + len(self._in_use)
            if total_connections < self.pool_size:
                conn = await self._create_connection()
                self._in_use.add(conn)
                return conn
        
        # Check timeout
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed >= self.timeout:
            raise TimeoutError(f"Could not acquire connection within {self.timeout}s")
        
        # Wait and retry
        await asyncio.sleep(0.1)
```

**Explanation:**
- Tries to get an existing connection from the pool
- Validates connection with `SELECT 1` before returning
- Creates new connection if pool isn't full
- Raises `TimeoutError` if no connection available within timeout

### 7.3 Connection Release

```python
async def release(self, conn: aiosqlite.Connection) -> None:
    """Release a connection back to the pool."""
    async with self._lock:
        if conn in self._in_use:
            self._in_use.remove(conn)
            if not self._closed:
                self._pool.append(conn)  # Return to pool
            else:
                await conn.close()       # Pool closed, close connection
```

### 7.4 Context Manager Usage

```python
@asynccontextmanager
async def db_connection():
    """Context manager for database connections."""
    conn = await get_db()
    try:
        yield conn
    finally:
        await release_db(conn)

# Usage example
async with db_connection() as db:
    result = await db.execute("SELECT * FROM logs_stream LIMIT 10")
    rows = await result.fetchall()
```

---

## 8. Schema Migrations (migrations.py)

**File:** `apps/api/app/storage/migrations.py`

### Purpose
Manages database schema versioning and migrations.

### Schema Version Tracking

```python
CURRENT_SCHEMA_VERSION = 2

async def get_schema_version(db: aiosqlite.Connection) -> int:
    """Get current schema version from database."""
    try:
        cursor = await db.execute("SELECT version FROM schema_version LIMIT 1")
        row = await cursor.fetchone()
        return row[0] if row else 0
    except:
        return 0  # Table doesn't exist

async def set_schema_version(db: aiosqlite.Connection, version: int) -> None:
    """Set schema version in database."""
    await db.execute("DELETE FROM schema_version")
    await db.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
    await db.commit()
```

### Schema V2 Tables

```python
SCHEMA_V2 = """
-- Log templates extracted by Drain algorithm
CREATE TABLE IF NOT EXISTS log_templates (
    tenant_id TEXT DEFAULT 'default',
    service_name TEXT NOT NULL,
    template_hash INTEGER NOT NULL,       -- xxhash64 of template
    template_text TEXT NOT NULL,          -- Template with <*> placeholders
    first_seen_utc TEXT,
    last_seen_utc TEXT,
    embedding_state TEXT DEFAULT 'none',  -- none|pending|done
    embedding_model TEXT,
    PRIMARY KEY (tenant_id, service_name, template_hash)
);

-- Individual log entries
CREATE TABLE IF NOT EXISTS logs_stream (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT DEFAULT 'default',
    service_name TEXT NOT NULL,
    environment TEXT DEFAULT 'production',
    timestamp_utc TEXT NOT NULL,
    ingest_timestamp_utc TEXT,
    severity INTEGER DEFAULT 1,           -- 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR, 4=CRITICAL
    host TEXT,
    template_hash INTEGER,                -- FK to log_templates
    parameters_json TEXT,                 -- JSON array of extracted parameters
    trace_id TEXT,
    span_id TEXT,
    attributes_json TEXT,                 -- Additional metadata
    body_raw TEXT NOT NULL                -- Original log message
);

-- Vector embeddings for semantic search
CREATE TABLE IF NOT EXISTS template_vectors (
    tenant_id TEXT DEFAULT 'default',
    service_name TEXT NOT NULL,
    template_hash INTEGER NOT NULL,
    faiss_id INTEGER NOT NULL,            -- ID in FAISS index
    vector_b64 TEXT,                      -- Base64-encoded vector
    PRIMARY KEY (tenant_id, service_name, template_hash)
);

-- Track ingested files to prevent duplicates
CREATE TABLE IF NOT EXISTS ingested_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_size INTEGER,
    file_hash TEXT,                       -- MD5/SHA hash for deduplication
    status TEXT DEFAULT 'pending',        -- pending|processing|completed|failed
    lines_processed INTEGER DEFAULT 0,
    events_inserted INTEGER DEFAULT 0,
    created_at TEXT,
    completed_at TEXT
);

-- Schema version table
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs_stream(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_logs_severity ON logs_stream(severity);
CREATE INDEX IF NOT EXISTS idx_logs_tenant_time ON logs_stream(tenant_id, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_logs_service_time ON logs_stream(service_name, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_logs_template ON logs_stream(template_hash);
"""
```

**Index Strategy:**
- `idx_logs_timestamp`: Fast time-range queries
- `idx_logs_severity`: Filter by severity level
- `idx_logs_tenant_time`: Multi-tenant time queries
- `idx_logs_service_time`: Service-specific time queries
- `idx_logs_template`: Find logs by template

---

## 9. Logs Repository (logs_repo.py)

**File:** `apps/api/app/storage/logs_repo.py`

### Purpose
Data access layer for log entries. Handles CRUD operations and complex queries.

### Class: LogsRepo

```python
class LogsRepo:
    """Repository for log entries in logs_stream table."""
    
    def __init__(self, db: aiosqlite.Connection):
        self.db = db
```

### 9.1 Query Logs with Filters

```python
async def query_logs(
    self,
    tenant_id: str,
    from_time: datetime,
    to_time: datetime,
    service_name: Optional[str] = None,
    severity_min: Optional[int] = None,
    template_hash: Optional[int] = None,
    trace_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Tuple[List[dict], int, bool]:
    """
    Query logs with filters.
    
    Returns:
        Tuple of (logs, total_count, has_more)
    """
    # Build dynamic WHERE clause
    conditions = ["tenant_id = ?", "timestamp_utc >= ?", "timestamp_utc <= ?"]
    params = [tenant_id, from_time.isoformat(), to_time.isoformat()]
    
    if service_name:
        conditions.append("service_name = ?")
        params.append(service_name)
    
    if severity_min is not None:
        conditions.append("severity >= ?")
        params.append(severity_min)
    
    if template_hash is not None:
        conditions.append("template_hash = ?")
        params.append(template_hash)
    
    if trace_id:
        conditions.append("trace_id = ?")
        params.append(trace_id)
    
    where_clause = " AND ".join(conditions)
    
    # Count total matching rows
    count_sql = f"SELECT COUNT(*) FROM logs_stream WHERE {where_clause}"
    cursor = await self.db.execute(count_sql, params)
    row = await cursor.fetchone()
    total = row[0] if row else 0
    
    # Fetch page of results
    query_sql = f"""
        SELECT * FROM logs_stream 
        WHERE {where_clause}
        ORDER BY timestamp_utc DESC
        LIMIT ? OFFSET ?
    """
    cursor = await self.db.execute(query_sql, params + [limit, offset])
    rows = await cursor.fetchall()
    
    logs = [dict(row) for row in rows]
    has_more = offset + len(logs) < total
    
    return logs, total, has_more
```

**Explanation:**
- Builds SQL dynamically based on provided filters
- Uses parameterized queries to prevent SQL injection
- Returns tuple with logs, total count, and pagination indicator

### 9.2 Batch Insert for Performance

```python
async def insert_batch(self, events: List[dict]) -> int:
    """
    Insert multiple log events efficiently.
    
    Args:
        events: List of normalized log event dicts
        
    Returns:
        Number of events inserted
    """
    if not events:
        return 0
    
    sql = """
        INSERT INTO logs_stream (
            tenant_id, service_name, environment, timestamp_utc,
            ingest_timestamp_utc, severity, host, template_hash,
            parameters_json, trace_id, span_id, attributes_json, body_raw
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    rows = [
        (
            e.get("tenant_id", "default"),
            e["service_name"],
            e.get("environment", "production"),
            e["timestamp_utc"],
            e.get("ingest_timestamp_utc"),
            e.get("severity", 1),
            e.get("host"),
            e.get("template_hash"),
            json.dumps(e.get("parameters", [])),
            e.get("trace_id"),
            e.get("span_id"),
            json.dumps(e.get("attributes", {})),
            e["body_raw"],
        )
        for e in events
    ]
    
    await self.db.executemany(sql, rows)
    return len(rows)
```

**Explanation:**
- Uses `executemany()` for batch insert (much faster than individual inserts)
- Converts lists/dicts to JSON strings for storage
- Returns count of inserted rows

### 9.3 Quick Statistics

```python
async def get_quick_stats(self, tenant_id: str) -> dict:
    """
    Get quick statistics without expensive COUNT(*).
    Uses table metadata for estimates.
    """
    # Get estimated row count from sqlite_stat1
    cursor = await self.db.execute(
        "SELECT stat FROM sqlite_stat1 WHERE tbl = 'logs_stream' LIMIT 1"
    )
    row = await cursor.fetchone()
    
    if row and row[0]:
        # Parse stat string: "nrow ..."
        parts = row[0].split()
        estimated_count = int(parts[0]) if parts else 0
    else:
        # Fallback to actual count (slower)
        cursor = await self.db.execute("SELECT COUNT(*) FROM logs_stream")
        row = await cursor.fetchone()
        estimated_count = row[0] if row else 0
    
    # Get template count
    cursor = await self.db.execute("SELECT COUNT(*) FROM log_templates")
    row = await cursor.fetchone()
    template_count = row[0] if row else 0
    
    # Get unique services
    cursor = await self.db.execute("SELECT COUNT(DISTINCT service_name) FROM logs_stream")
    row = await cursor.fetchone()
    service_count = row[0] if row else 0
    
    return {
        "logs_estimated": estimated_count,
        "templates": template_count,
        "services": service_count,
        "is_estimated": True
    }
```

**Explanation:**
- Uses SQLite statistics table for fast row count estimation
- Avoids expensive `COUNT(*)` on large tables
- Returns approximate counts for UI display

---

## 10. Templates Repository (templates_repo.py)

**File:** `apps/api/app/storage/templates_repo.py`

### Purpose
Manages log template storage and retrieval.

### 10.1 Upsert Template

```python
async def upsert_template(
    self,
    tenant_id: str,
    service_name: str,
    template_hash: int,
    template_text: str,
) -> bool:
    """
    Insert or update a template.
    
    Returns:
        True if new template, False if existing
    """
    now = datetime.utcnow().isoformat()
    
    # Try insert first (most common case)
    try:
        await self.db.execute(
            """
            INSERT INTO log_templates 
            (tenant_id, service_name, template_hash, template_text, first_seen_utc, last_seen_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (tenant_id, service_name, template_hash, template_text, now, now)
        )
        return True
    except aiosqlite.IntegrityError:
        # Template exists, update last_seen
        await self.db.execute(
            """
            UPDATE log_templates 
            SET last_seen_utc = ?
            WHERE tenant_id = ? AND service_name = ? AND template_hash = ?
            """,
            (now, tenant_id, service_name, template_hash)
        )
        return False
```

**Explanation:**
- Uses INSERT-first approach (optimistic)
- On duplicate key, updates `last_seen_utc` timestamp
- Returns boolean indicating if template was new

### 10.2 Get Top Templates

```python
async def get_top_templates(
    self,
    tenant_id: str,
    service_name: str,
    from_time: datetime,
    to_time: datetime,
    limit: int = 20,
) -> List[dict]:
    """Get most frequent templates with log counts."""
    
    sql = """
        SELECT 
            t.template_hash,
            t.template_text,
            t.first_seen_utc,
            t.last_seen_utc,
            COUNT(l.id) as log_count
        FROM log_templates t
        LEFT JOIN logs_stream l ON t.template_hash = l.template_hash 
            AND l.timestamp_utc >= ? AND l.timestamp_utc <= ?
        WHERE t.tenant_id = ? AND t.service_name = ?
        GROUP BY t.template_hash
        ORDER BY log_count DESC
        LIMIT ?
    """
    
    cursor = await self.db.execute(
        sql, 
        (from_time.isoformat(), to_time.isoformat(), tenant_id, service_name, limit)
    )
    rows = await cursor.fetchall()
    
    return [
        {
            "template_hash": str(row["template_hash"]),  # Convert to string for JS
            "template_text": row["template_text"],
            "first_seen": row["first_seen_utc"],
            "last_seen": row["last_seen_utc"],
            "count": row["log_count"],
        }
        for row in rows
    ]
```

**Explanation:**
- Joins templates with logs to get occurrence counts
- Filters by time range and service
- Returns template hashes as strings (JavaScript can't handle 64-bit integers)






---

*End of Part 2. Continue with Part 3 for Parsers & Log Processing.*

---







# Part 3: Parsers & Log Processing

## 11. Drain Template Miner (drain_miner.py)

**File:** `apps/api/app/parsers/drain_miner.py`

### Purpose
Implements the Drain algorithm for log template extraction. Converts variable log messages into templates with placeholders.

### What is Drain?
Drain is a log parsing algorithm that:
1. Groups similar log messages into clusters
2. Extracts common patterns (templates)
3. Identifies variable parts (parameters)

**Example:**
```
Input:  "User john logged in from 192.168.1.1"
Input:  "User alice logged in from 10.0.0.5"
Output: "User <*> logged in from <*>"
Params: ["john", "192.168.1.1"] and ["alice", "10.0.0.5"]
```

### Class: DrainMiner

```python
class DrainMiner:
    """
    Wrapper around Drain3 for log template mining.
    Extracts templates with <*> placeholders and parameter values.
    """
    
    def __init__(self, depth: int = 4, sim_th: float = 0.4, max_children: int = 100):
        """
        Initialize Drain miner.
        
        Args:
            depth: Depth of prefix tree (affects grouping)
            sim_th: Similarity threshold for clustering (0.0-1.0)
            max_children: Max children per tree node
        """
        self.config = TemplateMinerConfig()
        self.config.drain_depth = depth
        self.config.drain_sim_th = sim_th
        self.config.drain_max_children = max_children
        self.config.drain_extra_delimiters = ["_", ":", "=", "/", "\\", "[", "]", "(", ")"]
        self.config.mask_prefix = "<"
        self.config.mask_suffix = ">"
        
        self.miner = TemplateMiner(config=self.config)
```

**Parameter Explanations:**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `depth` | 4 | Tree depth - higher = more specific templates |
| `sim_th` | 0.4 | Similarity threshold - lower = more general templates |
| `max_children` | 100 | Max clusters per node |
| `extra_delimiters` | Various | Characters to split tokens on |

### 11.1 Processing Log Messages

```python
def add_log_message(self, message: str) -> Tuple[str, List[str], int]:
    """
    Process a log message and extract template.
    
    Args:
        message: Raw log message
        
    Returns:
        Tuple of (template_text, parameters, cluster_id)
    """
    if not message or not message.strip():
        return ("<*>", [message or ""], 0)
    
    # Process through Drain
    result = self.miner.add_log_message(message)
    
    if result is None:
        return ("<*>", [message], 0)
    
    # Handle both old API (object) and new API (dict)
    if isinstance(result, dict):
        template = result.get("template_mined", message)
        cluster_id = result.get("cluster_id", 0)
    else:
        template = result.get_template()
        cluster_id = result.cluster_id
    
    # Extract parameters
    parameters = self._extract_parameters(message, template)
    
    return (template, parameters, cluster_id)
```

**Explanation:**
- Handles edge cases (empty, None)
- Supports both old and new Drain3 API versions
- Extracts variable parameters from the original message

### 11.2 Parameter Extraction

```python
def _extract_parameters(self, message: str, template: str) -> List[str]:
    """
    Extract parameter values from message using template.
    
    Example:
        message: "User john logged in from 192.168.1.1"
        template: "User <*> logged in from <*>"
        returns: ["john", "192.168.1.1"]
    """
    if "<*>" not in template:
        return []
    
    # Split template into static parts around <*>
    parts = template.split("<*>")
    # parts = ["User ", " logged in from ", ""]
    
    parameters = []
    remaining = message
    
    for i, part in enumerate(parts[:-1]):
        # Find the static part in the remaining message
        if part:
            idx = remaining.find(part)
            if idx != -1:
                remaining = remaining[idx + len(part):]
        
        # Find where the next static part begins
        next_part = parts[i + 1]
        if next_part:
            idx = remaining.find(next_part)
            if idx != -1:
                param = remaining[:idx]  # Everything before next static part
                parameters.append(param)
                remaining = remaining[idx:]
            else:
                parameters.append(remaining)
                remaining = ""
        else:
            # No more static parts, rest is parameter
            parameters.append(remaining)
            remaining = ""
    
    return parameters
```

### 11.3 Template Hashing

```python
def compute_template_hash(service_name: str, template_text: str) -> int:
    """
    Compute deterministic 64-bit hash for a template.
    
    Uses xxhash64 for speed and good distribution.
    Returns signed 64-bit integer for SQLite compatibility.
    """
    # Normalize template
    normalized = template_text.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)  # Collapse whitespace
    
    # Create unique string combining service and template
    unique = f"{service_name.lower()}|{normalized}"
    
    # Compute xxhash64
    unsigned_hash = xxhash.xxh64(unique.encode('utf-8')).intdigest()
    
    # Convert to signed 64-bit (SQLite INTEGER is signed)
    if unsigned_hash >= 2**63:
        return unsigned_hash - 2**64
    return unsigned_hash
```

**Why xxhash?**
- Very fast (faster than MD5, SHA)
- Good distribution (few collisions)
- 64-bit output fits in SQLite INTEGER

### 11.4 Global Miner Cache

```python
# Cache miners per service to maintain state
_miners: dict[str, DrainMiner] = {}

def get_miner(service_name: str) -> DrainMiner:
    """Get or create a Drain miner for a service."""
    if service_name not in _miners:
        _miners[service_name] = DrainMiner()
    return _miners[service_name]

def mine_template(message: str, service_name: str) -> Tuple[str, List[str], int]:
    """Convenience function to mine a single message."""
    miner = get_miner(service_name)
    template, params, cluster_id = miner.add_log_message(message)
    template_hash = compute_template_hash(service_name, template)
    return template, params, template_hash
```

**Explanation:**
- Each service gets its own miner (different log patterns)
- Miners are cached to maintain learned patterns
- `mine_template()` is the main entry point

---

## 12. Text Parser (text_parser.py)

**File:** `apps/api/app/parsers/text_parser.py`

### Purpose
Parses unstructured text log files, extracting timestamps, severity, and message content.

### Main Function: parse_text_line

```python
def parse_text_line(
    line: str,
    default_service: str = "unknown",
    line_number: int = 0,
) -> Optional[dict]:
    """
    Parse a plain text log line.
    
    Attempts to extract:
    - Timestamp (various formats)
    - Severity level
    - Host/source
    - Log message body
    
    Returns:
        Parsed log dict or None if unparseable
    """
    if not line or not line.strip():
        return None
    
    line = line.strip()
    result = {
        "body_raw": line,
        "service_name": default_service,
        "severity": 1,  # Default INFO
        "timestamp_utc": None,
        "host": None,
    }
    
    # Try to extract timestamp
    timestamp = extract_timestamp(line)
    if timestamp:
        result["timestamp_utc"] = timestamp
    
    # Try to extract severity
    severity = extract_severity(line)
    if severity is not None:
        result["severity"] = severity
    
    # Try to extract host
    host = extract_host(line)
    if host:
        result["host"] = host
    
    return result
```

### 12.1 Timestamp Extraction

```python
# Regex patterns for common timestamp formats
TIMESTAMP_PATTERNS = [
    # ISO 8601: 2024-01-15T10:30:00Z
    (r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)', 
     "%Y-%m-%dT%H:%M:%S"),
    
    # Common log: Jan 15 10:30:00
    (r'([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})',
     "%b %d %H:%M:%S"),
    
    # Apache: [15/Jan/2024:10:30:00 +0000]
    (r'\[(\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4})\]',
     "%d/%b/%Y:%H:%M:%S %z"),
    
    # Epoch milliseconds: 1705312200000
    (r'^(\d{13})(?:\s|$)', "epoch_ms"),
    
    # Epoch seconds: 1705312200
    (r'^(\d{10})(?:\s|$)', "epoch_s"),
]

def extract_timestamp(line: str) -> Optional[str]:
    """Extract and normalize timestamp from log line."""
    for pattern, format_str in TIMESTAMP_PATTERNS:
        match = re.search(pattern, line)
        if match:
            ts_str = match.group(1)
            try:
                if format_str == "epoch_ms":
                    dt = datetime.fromtimestamp(int(ts_str) / 1000, tz=timezone.utc)
                elif format_str == "epoch_s":
                    dt = datetime.fromtimestamp(int(ts_str), tz=timezone.utc)
                else:
                    dt = datetime.strptime(ts_str, format_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat()
            except (ValueError, OSError):
                continue
    return None
```

### 12.2 Severity Extraction

```python
# Severity keywords and their levels
SEVERITY_KEYWORDS = {
    # Level 0: DEBUG
    "debug": 0, "trace": 0, "verbose": 0,
    
    # Level 1: INFO
    "info": 1, "information": 1, "notice": 1,
    
    # Level 2: WARNING
    "warn": 2, "warning": 2, "caution": 2,
    
    # Level 3: ERROR
    "error": 3, "err": 3, "failure": 3, "failed": 3,
    
    # Level 4: CRITICAL
    "critical": 4, "fatal": 4, "emergency": 4, "emerg": 4, "panic": 4,
}

def extract_severity(line: str) -> Optional[int]:
    """Extract severity level from log line."""
    line_lower = line.lower()
    
    # Check for bracketed severity: [ERROR], [WARN], etc.
    bracket_match = re.search(r'\[([A-Z]+)\]', line)
    if bracket_match:
        level = bracket_match.group(1).lower()
        if level in SEVERITY_KEYWORDS:
            return SEVERITY_KEYWORDS[level]
    
    # Check for severity keywords in the line
    for keyword, level in SEVERITY_KEYWORDS.items():
        # Match whole word only
        if re.search(rf'\b{keyword}\b', line_lower):
            return level
    
    return None
```

---

## 13. JSONL Parser (jsonl_parser.py)

**File:** `apps/api/app/parsers/jsonl_parser.py`

### Purpose
Parses JSON Lines format (one JSON object per line).

```python
def parse_jsonl_line(line: str, default_service: str = "unknown") -> Optional[dict]:
    """
    Parse a JSON Lines log entry.
    
    Handles various JSON log formats:
    - OpenTelemetry format
    - Bunyan format
    - Generic JSON logs
    """
    if not line or not line.strip():
        return None
    
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    
    if not isinstance(data, dict):
        return None
    
    # Map common field names to our schema
    result = {
        "body_raw": line,
        "service_name": default_service,
        "severity": 1,
        "timestamp_utc": None,
        "host": None,
        "trace_id": None,
        "span_id": None,
        "attributes": {},
    }
    
    # Timestamp field mapping
    for field in ["timestamp", "ts", "time", "@timestamp", "datetime", "date"]:
        if field in data:
            result["timestamp_utc"] = normalize_timestamp(data[field])
            break
    
    # Severity field mapping
    for field in ["severity", "level", "log_level", "loglevel", "sev"]:
        if field in data:
            result["severity"] = normalize_severity(data[field])
            break
    
    # Message field mapping
    for field in ["message", "msg", "body", "log", "text", "content"]:
        if field in data:
            result["body_raw"] = str(data[field])
            break
    
    # Service name mapping
    for field in ["service", "service_name", "app", "application", "component"]:
        if field in data:
            result["service_name"] = str(data[field])
            break
    
    # Trace context
    result["trace_id"] = data.get("trace_id") or data.get("traceId")
    result["span_id"] = data.get("span_id") or data.get("spanId")
    
    # Store remaining fields as attributes
    known_fields = {"timestamp", "ts", "time", "severity", "level", "message", "msg", ...}
    result["attributes"] = {k: v for k, v in data.items() if k not in known_fields}
    
    return result
```

---

## 14. Normalizer (normalize.py)

**File:** `apps/api/app/parsers/normalize.py`

### Purpose
Normalizes parsed log events to a consistent schema before storage.

```python
def normalize_event(
    parsed: dict,
    tenant_id: str = "default",
    service_name: Optional[str] = None,
    environment: str = "production",
) -> NormalizedLogEvent:
    """
    Normalize a parsed log event to standard schema.
    
    Args:
        parsed: Raw parsed dict from parser
        tenant_id: Tenant identifier
        service_name: Override service name
        environment: Deployment environment
        
    Returns:
        NormalizedLogEvent with all fields populated
    """
    # Generate timestamp if missing
    timestamp = parsed.get("timestamp_utc")
    if not timestamp:
        timestamp = datetime.utcnow().isoformat()
    
    # Normalize severity to 0-4 range
    severity = parsed.get("severity", 1)
    if isinstance(severity, str):
        severity = SEVERITY_KEYWORDS.get(severity.lower(), 1)
    severity = max(0, min(4, severity))  # Clamp to 0-4
    
    # Extract template and parameters
    body = parsed.get("body_raw", "")
    svc = service_name or parsed.get("service_name", "unknown")
    template, params, template_hash = mine_template(body, svc)
    
    return NormalizedLogEvent(
        tenant_id=tenant_id,
        service_name=svc,
        environment=environment,
        timestamp_utc=timestamp,
        ingest_timestamp_utc=datetime.utcnow().isoformat(),
        severity=severity,
        host=parsed.get("host"),
        template_hash=template_hash,
        template_text=template,
        parameters=params,
        trace_id=parsed.get("trace_id"),
        span_id=parsed.get("span_id"),
        attributes=parsed.get("attributes", {}),
        body_raw=body,
    )
```

### Normalization Steps:
1. **Timestamp**: Use parsed timestamp or current time
2. **Severity**: Convert to 0-4 integer range
3. **Template Mining**: Extract template and parameters via Drain
4. **Hash Generation**: Compute template hash for grouping
5. **Metadata**: Capture trace ID, span ID, attributes




---

*End of Part 3. Continue with Part 4 for Machine Learning Models.*

---





# Part 4: Machine Learning Models

## 15. Anomaly Detector (anomaly_detector.py)

**File:** `apps/api/app/ml/anomaly_detector.py`

### Purpose
Detects anomalous log patterns using multiple ML techniques:
1. **Isolation Forest** - Multivariate anomaly detection
2. **Statistical Methods** - Z-score and IQR outlier detection
3. **DBSCAN** - Density-based clustering

### Data Classes

```python
@dataclass
class AnomalyResult:
    """Result of anomaly detection analysis"""
    is_anomaly: bool
    anomaly_score: float          # 0-1, higher = more anomalous
    anomaly_type: str             # 'frequency', 'pattern', 'severity', 'temporal'
    confidence: float             # Model confidence
    explanation: str              # Human-readable explanation
    contributing_factors: List[Dict[str, Any]] = field(default_factory=list)
    related_templates: List[str] = field(default_factory=list)
```

### 15.1 Isolation Forest Implementation

```python
class IsolationForestDetector:
    """
    Isolation Forest implementation for log anomaly detection.
    
    HOW IT WORKS:
    - Randomly partitions the data space using binary trees
    - Anomalies are isolated quickly (short path length)
    - Normal points require many partitions to isolate
    
    INTUITION: Anomalies are "few and different", so they're
    easier to separate from the rest of the data.
    """
    
    def __init__(self, n_estimators: int = 100, contamination: float = 0.1):
        self.n_estimators = n_estimators  # Number of trees
        self.contamination = contamination  # Expected fraction of anomalies
        self.trees: List[Dict] = []
        self.threshold: float = 0.5
        self.trained = False
```

#### Building Isolation Trees

```python
def _build_tree(self, X: np.ndarray, max_depth: int) -> Dict:
    """Build a single isolation tree recursively."""
    n_samples, n_features = X.shape
    
    # Base case: leaf node
    if n_samples <= 1 or max_depth <= 0:
        return {"type": "leaf", "size": n_samples}
    
    # Random feature and split point
    feature_idx = np.random.randint(0, n_features)
    feature_values = X[:, feature_idx]
    
    # Can't split if all values are the same
    if feature_values.min() == feature_values.max():
        return {"type": "leaf", "size": n_samples}
    
    # Random split value between min and max
    split_value = np.random.uniform(feature_values.min(), feature_values.max())
    
    # Partition data
    left_mask = X[:, feature_idx] < split_value
    right_mask = ~left_mask
    
    return {
        "type": "split",
        "feature": feature_idx,
        "threshold": split_value,
        "left": self._build_tree(X[left_mask], max_depth - 1),
        "right": self._build_tree(X[right_mask], max_depth - 1)
    }
```

**Algorithm Explanation:**
1. Pick a random feature (column)
2. Pick a random split value in that feature's range
3. Partition data into left (< split) and right (>= split)
4. Recurse until leaf (1 sample) or max depth

#### Training the Forest

```python
def fit(self, X: np.ndarray) -> "IsolationForestDetector":
    """Train the Isolation Forest on log feature data."""
    n_samples = X.shape[0]
    
    # Normalize features (zero mean, unit variance)
    self.feature_means = X.mean(axis=0)
    self.feature_stds = X.std(axis=0) + 1e-8  # Avoid division by zero
    X_normalized = (X - self.feature_means) / self.feature_stds
    
    # Calculate max depth based on subsample size
    max_depth = int(np.ceil(np.log2(max(n_samples, 2))))
    
    # Build trees with subsampling (256 samples per tree)
    sample_size = min(256, n_samples)
    self.trees = []
    
    for _ in range(self.n_estimators):
        # Random subsample
        indices = np.random.choice(n_samples, size=sample_size, replace=False)
        tree = self._build_tree(X_normalized[indices], max_depth)
        self.trees.append(tree)
    
    # Calculate threshold based on contamination
    scores = self._score_samples(X_normalized)
    self.threshold = np.percentile(scores, 100 * (1 - self.contamination))
    
    self.trained = True
    return self
```

#### Calculating Path Length (Anomaly Score)

```python
def _path_length(self, x: np.ndarray, tree: Dict, current_depth: int = 0) -> float:
    """Calculate path length for a single sample."""
    if tree["type"] == "leaf":
        n = tree["size"]
        if n <= 1:
            return current_depth
        # Average path length for external nodes (Euler constant approximation)
        return current_depth + 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
    
    # Traverse tree
    if x[tree["feature"]] < tree["threshold"]:
        return self._path_length(x, tree["left"], current_depth + 1)
    return self._path_length(x, tree["right"], current_depth + 1)

def _score_samples(self, X: np.ndarray) -> np.ndarray:
    """Calculate anomaly scores for all samples."""
    avg_path_lengths = np.zeros(X.shape[0])
    
    for tree in self.trees:
        for i, x in enumerate(X):
            avg_path_lengths[i] += self._path_length(x, tree)
    
    avg_path_lengths /= len(self.trees)
    
    # Normalize to [0, 1] using formula from the original paper
    n = 256  # subsample size
    c_n = 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
    scores = 2 ** (-avg_path_lengths / c_n)
    
    return scores
```

**Score Interpretation:**
- Score close to 1: Anomaly (short path = easily isolated)
- Score close to 0: Normal (long path = hard to isolate)
- Score around 0.5: Indeterminate

### 15.2 Statistical Detector

```python
class StatisticalDetector:
    """
    Statistical anomaly detection using:
    - Z-score: Measures standard deviations from mean
    - Modified Z-score: Uses Median Absolute Deviation (robust to outliers)
    - IQR: Interquartile Range method
    """
    
    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        self.z_threshold = z_threshold      # Z-scores beyond this are anomalies
        self.iqr_multiplier = iqr_multiplier  # IQR multiplier for bounds
        self.baselines: Dict[str, Dict] = {}
        
    def fit(self, feature_dict: Dict[str, np.ndarray]) -> "StatisticalDetector":
        """Compute statistical baselines for each feature."""
        for feature_name, values in feature_dict.items():
            values = np.array(values)
            
            # Standard statistics
            mean = np.mean(values)
            std = np.std(values) + 1e-8
            median = np.median(values)
            mad = np.median(np.abs(values - median)) + 1e-8  # Median Absolute Deviation
            
            # IQR statistics
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            
            self.baselines[feature_name] = {
                "mean": mean,
                "std": std,
                "median": median,
                "mad": mad,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": q1 - self.iqr_multiplier * iqr,
                "upper_bound": q3 + self.iqr_multiplier * iqr,
            }
        
        return self
    
    def detect(self, feature_name: str, value: float) -> Tuple[bool, float, str]:
        """
        Detect if a value is anomalous for a feature.
        
        Returns:
            (is_anomaly, score, method)
        """
        if feature_name not in self.baselines:
            return False, 0.0, "unknown"
        
        baseline = self.baselines[feature_name]
        
        # Z-score method
        z_score = abs(value - baseline["mean"]) / baseline["std"]
        
        # Modified Z-score (robust to outliers)
        modified_z = 0.6745 * abs(value - baseline["median"]) / baseline["mad"]
        
        # IQR method
        iqr_violation = value < baseline["lower_bound"] or value > baseline["upper_bound"]
        
        # Combine methods
        if z_score > self.z_threshold or modified_z > self.z_threshold or iqr_violation:
            score = max(z_score, modified_z) / self.z_threshold
            method = "z-score" if z_score > modified_z else "modified-z"
            if iqr_violation:
                method = "iqr"
            return True, min(score, 1.0), method
        
        return False, z_score / self.z_threshold, "normal"
```

---

## 16. Log Classifier (log_classifier.py)

**File:** `apps/api/app/ml/log_classifier.py`

### Purpose
Classifies log entries by:
- **Category**: error, warning, security, performance, normal
- **Domain**: web_server, database, system, network, auth
- **Severity**: critical, high, medium, low, info

### Data Classes

```python
@dataclass
class ClassificationResult:
    """Result of log classification"""
    category: str                # 'error', 'warning', 'security', 'performance', 'normal'
    domain: str                  # 'web_server', 'database', 'system', 'network', etc.
    severity: str                # 'critical', 'high', 'medium', 'low', 'info'
    confidence: float            # 0-1 confidence score
    probabilities: Dict[str, float]  # Per-class probabilities
    extracted_entities: Dict[str, List[str]] = field(default_factory=dict)
```

### 16.1 TF-IDF Vectorizer

```python
class TFIDFVectorizer:
    """
    Custom TF-IDF implementation for log text vectorization.
    
    TF-IDF = Term Frequency × Inverse Document Frequency
    
    - TF: How often a term appears in a document
    - IDF: How rare/important a term is across all documents
    """
    
    def __init__(self, max_features: int = 5000, min_df: int = 2, ngram_range: Tuple = (1, 2)):
        self.max_features = max_features  # Max vocabulary size
        self.min_df = min_df              # Min document frequency
        self.ngram_range = ngram_range    # (1,2) = unigrams and bigrams
        self.vocabulary: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
```

#### Log-Specific Tokenization

```python
def _tokenize(self, text: str) -> List[str]:
    """Tokenize log text with log-specific preprocessing."""
    text = text.lower()
    
    # Normalize common log patterns to reduce vocabulary
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' IP_ADDR ', text)
    text = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', ' TIMESTAMP ', text)
    text = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', ' UUID ', text)
    text = re.sub(r'0x[0-9a-f]+', ' HEX_NUM ', text)
    text = re.sub(r'\b\d+\b', ' NUM ', text)
    text = re.sub(r'/[^\s]+', ' PATH ', text)
    
    # Split on non-alphanumeric
    tokens = re.findall(r'\b[a-z_][a-z_0-9]*\b', text)
    
    return tokens
```

**Why Normalize?**
- IP addresses like `192.168.1.1` and `10.0.0.5` become `IP_ADDR`
- Reduces vocabulary size dramatically
- Focuses on semantic content, not variable values

#### Computing TF-IDF

```python
def fit(self, documents: List[str]) -> "TFIDFVectorizer":
    """Fit the vectorizer on training documents."""
    df_counts = Counter()  # Document frequency for each term
    doc_count = len(documents)
    
    # Count in how many documents each term appears
    for doc in documents:
        tokens = self._tokenize(doc)
        ngrams = set(self._get_ngrams(tokens))  # Unique ngrams in this doc
        for ngram in ngrams:
            df_counts[ngram] += 1
    
    # Filter by min_df and select top features
    filtered_terms = [
        (term, count) for term, count in df_counts.items()
        if count >= self.min_df
    ]
    filtered_terms.sort(key=lambda x: x[1], reverse=True)
    filtered_terms = filtered_terms[:self.max_features]
    
    # Build vocabulary mapping: term -> index
    self.vocabulary = {term: idx for idx, (term, _) in enumerate(filtered_terms)}
    
    # Calculate IDF for each term
    # IDF = log((N + 1) / (df + 1)) + 1  (smoothed to avoid division by zero)
    self.idf = np.zeros(len(self.vocabulary))
    for term, idx in self.vocabulary.items():
        df = df_counts[term]
        self.idf[idx] = np.log((doc_count + 1) / (df + 1)) + 1
    
    return self

def transform(self, documents: List[str]) -> np.ndarray:
    """Transform documents to TF-IDF vectors."""
    vectors = np.zeros((len(documents), len(self.vocabulary)))
    
    for doc_idx, doc in enumerate(documents):
        tokens = self._tokenize(doc)
        ngrams = self._get_ngrams(tokens)
        tf = Counter(ngrams)  # Term frequency
        
        for term, count in tf.items():
            if term in self.vocabulary:
                idx = self.vocabulary[term]
                # TF with log normalization × IDF
                vectors[doc_idx, idx] = (1 + np.log(count)) * self.idf[idx]
    
    # L2 normalization (unit vectors for cosine similarity)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms
    
    return vectors
```

### 16.2 Naive Bayes Classifier

```python
class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes for text classification.
    
    Bayes' theorem: P(class|text) ∝ P(text|class) × P(class)
    
    Naive assumption: Words are conditionally independent given the class.
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.class_log_prior: Dict[str, float] = {}
        self.feature_log_prob: Dict[str, np.ndarray] = {}
        self.classes: List[str] = []
    
    def fit(self, X: np.ndarray, y: List[str]) -> "NaiveBayesClassifier":
        """Train classifier on TF-IDF features and labels."""
        self.classes = list(set(y))
        n_samples, n_features = X.shape
        
        for cls in self.classes:
            # Samples belonging to this class
            mask = np.array([label == cls for label in y])
            X_c = X[mask]
            
            # P(class) = count(class) / total
            self.class_log_prior[cls] = np.log(X_c.shape[0] / n_samples)
            
            # P(feature|class) with Laplace smoothing
            feature_count = X_c.sum(axis=0) + self.alpha
            total_count = feature_count.sum()
            self.feature_log_prob[cls] = np.log(feature_count / total_count)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate probability for each class."""
        log_probs = {}
        
        for cls in self.classes:
            # log P(class|features) = log P(class) + sum(log P(feature|class))
            log_probs[cls] = self.class_log_prior[cls] + (X @ self.feature_log_prob[cls])
        
        # Convert to probabilities via softmax
        # ... (normalization code)
        return log_probs
```

---

## 17. Security Threat Detector (security_threat_detector.py)

**File:** `apps/api/app/ml/security_threat_detector.py`

### Purpose
Detects security threats using:
1. **Markov Chains** - Model attack sequences
2. **Brute Force Detection** - Time-series analysis
3. **Injection Detection** - Pattern matching + ML
4. **Reconnaissance Detection** - Scan pattern analysis

### Data Classes

```python
@dataclass
class ThreatDetectionResult:
    """Result of security threat analysis"""
    is_threat: bool
    threat_score: float           # 0-1, higher = more dangerous
    threat_type: str              # 'brute_force', 'injection', 'reconnaissance', 'dos'
    confidence: float
    severity: str                 # 'critical', 'high', 'medium', 'low'
    attack_indicators: List[Dict[str, Any]] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
```

### 17.1 Attack Sequence Detector (Markov Chains)

```python
class AttackSequenceDetector:
    """
    Detects attack patterns using Markov chain sequence analysis.
    
    INTUITION: Attacks follow predictable sequences:
    - Brute force: FAIL -> FAIL -> FAIL -> SUCCESS
    - Reconnaissance: REQUEST -> REQUEST -> REQUEST (many different paths)
    
    Normal traffic has different transition patterns.
    """
    
    def __init__(self, order: int = 2):
        self.order = order  # Markov chain order (context size)
        # Transition counts: state -> next_event -> count
        self.normal_transitions: Dict[tuple, Counter] = defaultdict(Counter)
        self.attack_transitions: Dict[tuple, Counter] = defaultdict(Counter)
```

#### Event Signature Extraction

```python
def _get_event_signature(self, log: Dict) -> str:
    """Extract a signature from a log event for sequence analysis."""
    components = []
    message = log.get("message", "").lower()
    
    # Classify event type
    if "login" in message or "auth" in message:
        components.append("AUTH")
    elif "failed" in message or "error" in message:
        components.append("FAIL")
    elif "success" in message:
        components.append("SUCCESS")
    elif "request" in message or "get" in message:
        components.append("REQUEST")
    elif "denied" in message or "blocked" in message:
        components.append("DENIED")
    else:
        components.append("OTHER")
    
    # Add source if available
    source = log.get("source", "")[:10].upper()
    if source:
        components.append(source)
    
    return "_".join(components)
```

#### Training on Normal vs Attack Sequences

```python
def fit(self, normal_logs: List[Dict], attack_logs: List[Dict]) -> "AttackSequenceDetector":
    """Train on labeled sequences."""
    # Learn normal transition probabilities
    if normal_logs:
        signatures = [self._get_event_signature(log) for log in normal_logs]
        for i in range(len(signatures) - self.order):
            state = tuple(signatures[i:i+self.order])  # Current context
            next_event = signatures[i+self.order]       # Next event
            self.normal_transitions[state][next_event] += 1
    
    # Learn attack transition probabilities
    if attack_logs:
        signatures = [self._get_event_signature(log) for log in attack_logs]
        for i in range(len(signatures) - self.order):
            state = tuple(signatures[i:i+self.order])
            next_event = signatures[i+self.order]
            self.attack_transitions[state][next_event] += 1
    
    return self

def score_sequence(self, logs: List[Dict]) -> float:
    """
    Score a sequence for attack likelihood.
    
    Uses log-likelihood ratio: P(sequence|attack) / P(sequence|normal)
    """
    signatures = [self._get_event_signature(log) for log in logs]
    
    normal_log_prob = 0.0
    attack_log_prob = 0.0
    
    for i in range(len(signatures) - self.order):
        state = tuple(signatures[i:i+self.order])
        next_event = signatures[i+self.order]
        
        # P(next|state) for normal model
        normal_prob = self._get_transition_prob(state, next_event, self.normal_transitions)
        normal_log_prob += np.log(normal_prob)
        
        # P(next|state) for attack model
        attack_prob = self._get_transition_prob(state, next_event, self.attack_transitions)
        attack_log_prob += np.log(attack_prob)
    
    # Convert to probability via sigmoid
    log_ratio = attack_log_prob - normal_log_prob
    return 1 / (1 + np.exp(-log_ratio))
```

### 17.2 Brute Force Detector

```python
class BruteForceDetector:
    """
    Detects brute force attacks using time-series analysis.
    
    Key indicators:
    - High failure rate from single source
    - Failures in rapid succession
    - Eventual success after many failures
    """
    
    def __init__(self, failure_threshold: int = 5, time_window_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.time_window_seconds = time_window_seconds
        # Track failures per source: source_ip -> deque of timestamps
        self.failure_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def add_event(self, log: Dict) -> Optional[ThreatDetectionResult]:
        """Process a log event and check for brute force."""
        # Extract source IP
        source_ip = self._extract_source_ip(log)
        if not source_ip:
            return None
        
        # Check if this is a failure event
        if not self._is_failure_event(log):
            return None
        
        # Record failure
        timestamp = datetime.fromisoformat(log.get("timestamp", datetime.now().isoformat()))
        self.failure_history[source_ip].append(timestamp)
        
        # Clean old entries outside time window
        cutoff = timestamp - timedelta(seconds=self.time_window_seconds)
        while self.failure_history[source_ip] and self.failure_history[source_ip][0] < cutoff:
            self.failure_history[source_ip].popleft()
        
        # Check if threshold exceeded
        failure_count = len(self.failure_history[source_ip])
        if failure_count >= self.failure_threshold:
            return ThreatDetectionResult(
                is_threat=True,
                threat_score=min(failure_count / (self.failure_threshold * 2), 1.0),
                threat_type="brute_force",
                confidence=0.85,
                severity="high" if failure_count > self.failure_threshold * 2 else "medium",
                attack_indicators=[{
                    "type": "repeated_failures",
                    "count": failure_count,
                    "source_ip": source_ip,
                    "time_window_seconds": self.time_window_seconds
                }],
                recommended_actions=[
                    f"Block IP {source_ip} temporarily",
                    "Enable CAPTCHA for this source",
                    "Review authentication logs",
                ]
            )
        
        return None
```

---

## 18. Training Pipeline (training_pipeline.py)

**File:** `apps/api/app/ml/training_pipeline.py`

### Purpose
Orchestrates training of all ML models from log data.

```python
class TrainingPipeline:
    """
    Full training pipeline for all ML models.
    
    Steps:
    1. Load log data from Logs/loghub directory
    2. Preprocess and label data
    3. Train each model (anomaly, classifier, security)
    4. Evaluate on held-out test set
    5. Save models to disk
    """
    
    async def run_full_pipeline(
        self,
        max_logs_per_source: int = 2000,
        train_ratio: float = 0.8,
    ) -> Dict[str, Any]:
        """Run complete training pipeline."""
        
        # Step 1: Load data
        logger.info("Loading log data...")
        logs = await self._load_training_data(max_logs_per_source)
        
        # Step 2: Split into train/test
        train_logs, test_logs = self._train_test_split(logs, train_ratio)
        
        # Step 3: Train models
        results = {}
        
        # Anomaly Detector
        logger.info("Training anomaly detector...")
        anomaly_detector = AnomalyDetector()
        await anomaly_detector.train(train_logs)
        results["anomaly"] = await self._evaluate_anomaly(anomaly_detector, test_logs)
        await anomaly_detector.save()
        
        # Log Classifier
        logger.info("Training log classifier...")
        classifier = LogClassifier()
        await classifier.train(train_logs)
        results["classifier"] = await self._evaluate_classifier(classifier, test_logs)
        await classifier.save()
        
        # Security Detector
        logger.info("Training security detector...")
        security = SecurityThreatDetector()
        await security.train(train_logs)
        results["security"] = await self._evaluate_security(security, test_logs)
        await security.save()
        
        return {
            "status": "completed",
            "logs_processed": len(logs),
            "train_size": len(train_logs),
            "test_size": len(test_logs),
            "results": results,
        }
```

### Model Persistence

```python
async def save(self) -> None:
    """Save model to disk."""
    model_path = Path(settings.models_dir) / "anomaly_detector.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, "wb") as f:
        pickle.dump({
            "isolation_forest": self.isolation_forest,
            "statistical": self.statistical,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
        }, f)

async def load(self) -> bool:
    """Load model from disk."""
    model_path = Path(settings.models_dir) / "anomaly_detector.pkl"
    
    if not model_path.exists():
        return False
    
    with open(model_path, "rb") as f:
        data = pickle.load(f)
        self.isolation_forest = data["isolation_forest"]
        self.statistical = data["statistical"]
        # ... restore other attributes
    
    return True
```

---

*End of Part 4. Continue with Part 5 for API Routes & Services.*

---

# Part 5: API Routes & Services

## 19. ML Routes (ml.py)

**File:** `apps/api/app/routes/ml.py`

### Purpose
Exposes ML model capabilities via REST API endpoints, including:
- Model training
- Anomaly detection
- Log classification
- Security threat detection
- AI chat with log search

### Request/Response Models

```python
class LogEntry(BaseModel):
    """Single log entry for analysis"""
    message: str
    timestamp: Optional[str] = None
    severity: Optional[str] = "INFO"
    source: Optional[str] = None

class BatchLogsRequest(BaseModel):
    """Batch of logs for analysis"""
    logs: List[LogEntry]

class AnomalyResponse(BaseModel):
    """Response from anomaly detection"""
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str
    confidence: float
    explanation: str
    contributing_factors: List[Dict[str, Any]] = []

class ChatRequest(BaseModel):
    """AI chat request"""
    message: str
    fast_mode: bool = True
    use_llm: bool = False
    context: List[str] = []
```

### 19.1 Training Endpoint

```python
@router.post("/train")
async def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Trigger training of all ML models.
    Runs in background and returns immediately.
    """
    pipeline = get_training_pipeline()
    
    # Check if models exist and force_retrain is False
    status = pipeline.get_status()
    if not request.force_retrain and all(status["models_exist"].values()):
        return {
            "status": "skipped",
            "message": "Models already exist. Use force_retrain=true to retrain.",
        }
    
    # Run training in background
    async def run_training():
        try:
            result = await pipeline.run_full_pipeline(
                max_logs_per_source=request.max_logs_per_source,
                train_ratio=request.train_ratio
            )
            logger.info(f"Training completed: {result}")
        except Exception as e:
            logger.error(f"Training failed: {e}")
    
    background_tasks.add_task(run_training)
    
    return {"status": "started", "message": "Training started in background"}
```

**Key Points:**
- Uses FastAPI's `BackgroundTasks` for async training
- Doesn't block the response - returns immediately
- Checks for existing models to avoid unnecessary retraining

### 19.2 AI Chat Endpoint

```python
@router.post("/chat")
async def ml_chat(request: ChatRequest):
    """
    Intelligent chat endpoint with multiple capabilities:
    1. Log search (when user asks for logs)
    2. Security Q&A (when user asks security questions)
    3. ML analysis (anomaly, classification, threat detection)
    4. LLM mode (full AI responses via Ollama)
    """
    message = request.message
    
    # Detect if this is a log search request
    if _is_log_search_request(message):
        return await _handle_log_search(message)
    
    # Run ML analysis on the message
    analysis = await _run_ml_analysis(message)
    
    # Generate response based on mode
    if request.use_llm:
        response = await _generate_llm_response(message, analysis, request.context)
    else:
        response = await _generate_fallback_response(message, analysis)
    
    return {
        "response": response,
        "analysis": analysis,
        "model_status": _get_model_status(),
        "suggestions": _get_follow_up_suggestions(message),
    }
```

### 19.3 Log Search Implementation

```python
def _is_log_search_request(message: str) -> bool:
    """Detect if user is asking for logs."""
    message_lower = message.lower()
    
    search_patterns = [
        r'\bshow\s+(me\s+)?(the\s+)?logs?\b',
        r'\bfind\s+(me\s+)?(the\s+)?logs?\b',
        r'\bsearch\s+(for\s+)?logs?\b',
        r'\blogs?\s+from\b',
        r'\bget\s+(me\s+)?(the\s+)?logs?\b',
    ]
    
    return any(re.search(p, message_lower) for p in search_patterns)

async def _handle_log_search(message: str) -> dict:
    """Handle log search requests."""
    # Extract search terms and service filter
    search_terms = _extract_search_terms(message)
    service_filter = _detect_service_filter(message)
    
    # Search database
    logs = await _search_logs_in_database(search_terms, limit=20, service_filter=service_filter)
    
    # Format results
    response = _format_log_search_results(logs, search_terms, service_filter)
    
    return {
        "response": response,
        "analysis": {"type": "log_search", "terms": search_terms},
        "model_status": _get_model_status(),
    }
```

### 19.4 Dynamic Service Detection

```python
def _detect_service_filter(message: str) -> Optional[str]:
    """
    Dynamically detect service name from user message.
    Uses regex to extract service names rather than hardcoded mappings.
    """
    message_lower = message.lower()
    
    # Pattern: "from <service>" or "in <service>" or "<service> logs"
    patterns = [
        r'\bfrom\s+(\w+)\b',
        r'\bin\s+(\w+)\b',
        r'\b(\w+)\s+logs?\b',
        r'\b(\w+)\s+service\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message_lower)
        if match:
            potential_service = match.group(1)
            # Validate it's not a common word
            common_words = {'the', 'my', 'our', 'all', 'some', 'error', 'warning'}
            if potential_service not in common_words:
                return potential_service
    
    return None
```

### 19.5 Semantic Search Term Mapping

```python
def _extract_search_terms(message: str) -> List[str]:
    """
    Extract search terms with semantic understanding.
    Maps user intent to actual log patterns.
    """
    message_lower = message.lower()
    terms = []
    
    # Semantic mappings: what user says -> what to search for
    semantic_mappings = {
        "incorrect password": ["authentication failed", "invalid password", "login failed"],
        "wrong password": ["authentication failed", "invalid password", "failed password"],
        "login failed": ["authentication failed", "login failure", "access denied"],
        "connection timeout": ["timeout", "connection refused", "timed out"],
        "server down": ["unreachable", "connection refused", "service unavailable"],
    }
    
    # Check for semantic matches
    for user_phrase, search_patterns in semantic_mappings.items():
        if user_phrase in message_lower:
            terms.extend(search_patterns)
    
    # Also extract quoted terms: "exact phrase"
    quoted = re.findall(r'"([^"]+)"', message)
    terms.extend(quoted)
    
    # Extract significant words (not stop words)
    stop_words = {'show', 'me', 'the', 'logs', 'from', 'find', 'search', 'for'}
    words = message_lower.split()
    terms.extend([w for w in words if w not in stop_words and len(w) > 2])
    
    return list(set(terms))  # Deduplicate
```

### 19.6 Log Search Database Query

```python
async def _search_logs_in_database(
    search_terms: List[str],
    limit: int = 20,
    service_filter: Optional[str] = None
) -> List[dict]:
    """Search logs in database matching criteria."""
    
    async with db_connection() as db:
        logs_repo = LogsRepo(db)
        
        # Build query
        conditions = ["1=1"]  # Always true base
        params = []
        
        # Add service filter
        if service_filter:
            conditions.append("LOWER(service_name) LIKE ?")
            params.append(f"%{service_filter.lower()}%")
        
        # Add text search for each term
        if search_terms:
            term_conditions = []
            for term in search_terms:
                term_conditions.append("LOWER(body_raw) LIKE ?")
                params.append(f"%{term.lower()}%")
            conditions.append(f"({' OR '.join(term_conditions)})")
        
        sql = f"""
            SELECT * FROM logs_stream
            WHERE {' AND '.join(conditions)}
            ORDER BY timestamp_utc DESC
            LIMIT ?
        """
        params.append(limit)
        
        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
```

### 19.7 Result Formatting with Severity Conversion

```python
def _format_log_search_results(
    logs: List[dict],
    search_terms: List[str],
    service_filter: Optional[str]
) -> str:
    """Format log search results as markdown."""
    
    # Severity number to name mapping
    severity_map = {0: "DEBUG", 1: "INFO", 2: "WARNING", 3: "ERROR", 4: "CRITICAL"}
    
    if not logs:
        return f"No logs found matching: {', '.join(search_terms)}"
    
    # Build response
    lines = [f"## 🔍 Log Search Results\n"]
    lines.append(f"Found **{len(logs)}** logs matching your query.\n")
    
    if service_filter:
        lines.append(f"**Service Filter:** `{service_filter}`\n")
    
    lines.append("| Time | Service | Severity | Message |")
    lines.append("|------|---------|----------|---------|")
    
    for log in logs:
        timestamp = log.get("timestamp_utc", "N/A")[:19]
        service = log.get("service_name", "unknown")
        severity_num = log.get("severity", 1)
        severity = severity_map.get(severity_num, str(severity_num))
        message = log.get("body_raw", "")[:80] + ("..." if len(log.get("body_raw", "")) > 80 else "")
        
        lines.append(f"| {timestamp} | {service} | {severity} | {message} |")
    
    return "\n".join(lines)
```

---

## 20. Ingest Service (ingest_service.py)

**File:** `apps/api/app/services/ingest_service.py`

### Purpose
Handles log ingestion from files and API. Optimized for high throughput with batch operations.

### Performance Constants

```python
BATCH_SIZE = 5000       # Events per batch insert
COMMIT_INTERVAL = 10000 # Events between commits
MAX_CONCURRENT_FILES = 3 # Parallel file processing
```

### Class: IngestService

```python
class IngestService:
    """Service for ingesting log data - optimized for performance."""
    
    def __init__(self):
        self._template_cache: Set[Tuple[str, str, int]] = set()
        self._progress_callback: Optional[ProgressCallback] = None
```

### 20.1 Folder Ingestion

```python
async def ingest_from_folder(
    self,
    folder_path: Optional[Path] = None,
    tenant_id: Optional[str] = None,
    skip_duplicates: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> IngestStats:
    """
    Ingest all log files from a folder.
    
    Features:
    - File hash tracking to skip duplicates
    - Progress callbacks for UI
    - Per-file commits for resilience
    """
    folder_path = folder_path or settings.logs_folder_resolved
    tenant_id = tenant_id or settings.tenant_id_default
    
    stats = IngestStats(files_processed=0, files_skipped=0, ...)
    
    # Discover log files
    files = discover_log_files(folder_path)
    
    db = await get_db()
    templates_repo = TemplatesRepo(db)
    logs_repo = LogsRepo(db)
    ingested_files_repo = IngestedFilesRepo(db)
    
    # Pre-load template cache for faster lookups
    await self._preload_template_cache(templates_repo, tenant_id)
    
    for filepath in files:
        # Check for duplicate files
        if skip_duplicates:
            file_hash = compute_file_hash(filepath)
            if await ingested_files_repo.get_ingested_file(file_hash):
                stats.files_skipped += 1
                continue
        
        # Ingest file
        file_stats = await self._ingest_file_fast(
            filepath=filepath,
            templates_repo=templates_repo,
            logs_repo=logs_repo,
            db=db,
        )
        
        stats.files_processed += 1
        stats.events_inserted += file_stats["events"]
        
        # COMMIT AFTER EACH FILE for resilience
        await db.commit()
    
    return stats
```

### 20.2 Fast File Ingestion

```python
async def _ingest_file_fast(
    self,
    filepath: Path,
    templates_repo: TemplatesRepo,
    logs_repo: LogsRepo,
    db: aiosqlite.Connection,
) -> dict:
    """Ingest a single file with optimizations."""
    
    # Infer service name from file path
    service_name = infer_service_name_from_path(filepath)
    file_format = get_file_format(filepath)
    
    events_batch = []
    templates_batch = []
    line_count = 0
    events_inserted = 0
    
    # Read and parse file
    async for line in read_file_lines(filepath):
        line_count += 1
        
        # Parse based on format
        if file_format == "jsonl":
            parsed = parse_jsonl_line(line, service_name)
        elif file_format == "csv":
            parsed = parse_csv_structured_line(line, service_name)
        else:
            parsed = parse_text_line(line, service_name)
        
        if not parsed:
            continue
        
        # Normalize to standard schema
        event = normalize_event(parsed, tenant_id=tenant_id, service_name=service_name)
        events_batch.append(event.__dict__)
        
        # Track new templates
        template_key = (tenant_id, service_name, event.template_hash)
        if template_key not in self._template_cache:
            self._template_cache.add(template_key)
            templates_batch.append({
                "tenant_id": tenant_id,
                "service_name": service_name,
                "template_hash": event.template_hash,
                "template_text": event.template_text,
            })
        
        # Batch insert when buffer is full
        if len(events_batch) >= BATCH_SIZE:
            # Insert templates first (for FK integrity)
            await templates_repo.upsert_batch(templates_batch)
            templates_batch = []
            
            # Insert events
            await logs_repo.insert_batch(events_batch)
            events_inserted += len(events_batch)
            events_batch = []
    
    # Insert remaining
    if templates_batch:
        await templates_repo.upsert_batch(templates_batch)
    if events_batch:
        await logs_repo.insert_batch(events_batch)
        events_inserted += len(events_batch)
    
    return {"lines": line_count, "events": events_inserted}
```

---

## 21. Chat Service (chat_service.py)

**File:** `apps/api/app/services/chat_service.py`

### Purpose
Handles AI chat logic including prompt construction and response generation.

### Few-Shot Examples

```python
# Loaded from few_shot_examples.json
FEW_SHOT_EXAMPLES = [
    {
        "user": "Is this log suspicious: Failed password for root from 192.168.1.1",
        "assistant": "Yes, this is suspicious. Multiple failed password attempts for the root user, especially from external IPs, is a common indicator of brute-force attacks..."
    },
    # ... more examples
]
```

### 21.1 Chat Message Handler

```python
class ChatService:
    """Handles AI chat interactions."""
    
    def __init__(self):
        self.ollama = get_ollama_client()
        self.few_shot = load_few_shot_examples()
    
    async def process_message(
        self,
        message: str,
        analysis: dict,
        context: List[str],
        use_llm: bool = False,
    ) -> str:
        """Process user message and generate response."""
        
        if use_llm:
            return await self._generate_llm_response(message, analysis, context)
        else:
            return await self._generate_fast_response(message, analysis)
    
    async def _generate_llm_response(
        self,
        message: str,
        analysis: dict,
        context: List[str],
    ) -> str:
        """Generate response using Ollama LLM."""
        
        # Build system prompt
        system_prompt = self._build_system_prompt(analysis)
        
        # Build messages with few-shot examples
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add few-shot examples
        for example in self.few_shot[:3]:
            messages.append({"role": "user", "content": example["user"]})
            messages.append({"role": "assistant", "content": example["assistant"]})
        
        # Add conversation context
        for ctx in context[-4:]:  # Last 4 messages
            messages.append({"role": "user", "content": ctx})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Call LLM
        response = await self.ollama.chat(messages, temperature=0.7)
        
        return response
    
    def _build_system_prompt(self, analysis: dict) -> str:
        """Build system prompt with ML analysis context."""
        return f"""You are LogMind AI, an expert log analysis assistant.

Current Analysis:
- Anomaly Score: {analysis.get('anomaly_score', 'N/A')}
- Classification: {analysis.get('category', 'N/A')}
- Threat Level: {analysis.get('threat_level', 'N/A')}

Your capabilities:
1. Analyze log patterns and detect anomalies
2. Identify security threats and attacks
3. Explain error messages and suggest fixes
4. Provide security best practices

Be concise and technical. Use markdown formatting."""
```

---

## 22. Ollama Client (ollama_client.py)

**File:** `apps/api/app/llm/ollama_client.py`

### Purpose
HTTP client for interacting with local Ollama LLM server.

### Class: OllamaClient

```python
class OllamaClient:
    """
    Client for Ollama local LLM API.
    Provides embeddings and chat completion.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        chat_model: Optional[str] = None,
        embed_model: Optional[str] = None,
        timeout: float = 300.0,  # 5 minutes for slow models
    ):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.chat_model = chat_model or settings.ollama_chat_model
        self.embed_model = embed_model or settings.ollama_embed_model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
```

### 22.1 Embedding Generation

```python
async def embed(self, text: str, model: Optional[str] = None) -> List[float]:
    """
    Generate embedding for text.
    
    Returns:
        Vector of floats (dimension depends on model)
    """
    model = model or self.embed_model
    
    client = await self._get_client()
    response = await client.post(
        "/api/embeddings",
        json={"model": model, "prompt": text}
    )
    response.raise_for_status()
    
    data = response.json()
    embedding = data.get("embedding")
    
    if embedding is None:
        raise OllamaError("No embedding in response")
    
    return embedding

async def embed_batch(self, texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts."""
    embeddings = []
    for text in texts:
        try:
            emb = await self.embed(text)
            embeddings.append(emb)
        except OllamaError:
            embeddings.append([])  # Empty on failure
    return embeddings
```

### 22.2 Chat Completion

```python
async def chat(
    self,
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.7,
    stream: bool = False,
) -> str:
    """
    Generate chat completion.
    
    Args:
        messages: List of {"role": "user/assistant/system", "content": "..."}
        model: Model to use
        system: System prompt
        temperature: Creativity (0 = deterministic, 1 = creative)
        stream: Whether to stream response
        
    Returns:
        Generated text response
    """
    model = model or self.chat_model
    
    # Build request
    request_body = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": temperature,
        }
    }
    
    if system:
        request_body["system"] = system
    
    try:
        client = await self._get_client()
        response = await client.post("/api/chat", json=request_body)
        response.raise_for_status()
        
        data = response.json()
        return data.get("message", {}).get("content", "")
        
    except httpx.TimeoutException:
        raise OllamaError("Request timed out. The model may be loading.")
    except httpx.ConnectError:
        raise OllamaConnectionError("Cannot connect to Ollama. Is it running?")
```

### 22.3 Health Check

```python
async def is_available(self) -> bool:
    """Check if Ollama is available and responding."""
    try:
        client = await self._get_client()
        response = await client.get("/api/tags")
        return response.status_code == 200
    except Exception:
        return False

async def list_models(self) -> List[str]:
    """List available models in Ollama."""
    try:
        client = await self._get_client()
        response = await client.get("/api/tags")
        response.raise_for_status()
        data = response.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []
```

### 22.4 Connection Management

```python
async def _get_client(self) -> httpx.AsyncClient:
    """Get or create HTTP client with connection pooling."""
    if self._client is None or self._client.is_closed:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            headers=headers,
        )
    return self._client

async def close(self) -> None:
    """Close HTTP client and release resources."""
    if self._client is not None:
        await self._client.aclose()
        self._client = None

# Global client instance
_ollama_client: Optional[OllamaClient] = None

def get_ollama_client() -> OllamaClient:
    """Get global Ollama client instance."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client

async def close_ollama_client() -> None:
    """Close global Ollama client on shutdown."""
    global _ollama_client
    if _ollama_client is not None:
        await _ollama_client.close()
        _ollama_client = None
```

---

*End of Part 5. Continue with Part 6 for Frontend Components.*

---

# Part 6: Frontend Components

## 23. API Client (api.ts)

**File:** `apps/ui/src/lib/api.ts`

### Purpose
TypeScript client for communicating with the FastAPI backend. Features:
- Automatic retry with exponential backoff
- Request deduplication
- Timeout handling
- Connection keepalive

### Configuration Constants

```typescript
const API_BASE = '/api';  // Proxied through Next.js

// Timeout configurations
const LOG_QUERY_TIMEOUT = 60000;   // 60s for log queries
const DEFAULT_TIMEOUT = 30000;     // 30s default
const FAST_TIMEOUT = 10000;        // 10s for cached endpoints

// Retry configuration
const MAX_RETRIES = 3;
const INITIAL_RETRY_DELAY = 500;   // 500ms initial delay
```

### 23.1 Retry with Exponential Backoff

```typescript
// Request deduplication cache
const pendingRequests = new Map<string, Promise<any>>();

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

const getRetryDelay = (attempt: number) => INITIAL_RETRY_DELAY * Math.pow(2, attempt);
// attempt 0: 500ms, attempt 1: 1000ms, attempt 2: 2000ms

async function fetchWithRetry<T>(
  url: string,
  options?: RequestInit & { timeout?: number; retries?: number; dedupe?: boolean }
): Promise<T> {
  const { 
    timeout = DEFAULT_TIMEOUT, 
    retries = MAX_RETRIES,
    dedupe = true,
    ...fetchOptions 
  } = options || {};

  // Deduplication: return existing promise for same GET request
  const cacheKey = `${fetchOptions.method || 'GET'}:${url}`;
  if (dedupe && (!fetchOptions.method || fetchOptions.method === 'GET')) {
    const pending = pendingRequests.get(cacheKey);
    if (pending) {
      return pending as Promise<T>;  // Return existing in-flight request
    }
  }

  const executeRequest = async (): Promise<T> => {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= retries; attempt++) {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      try {
        const response = await fetch(url, {
          ...fetchOptions,
          signal: controller.signal,
          headers: {
            'Content-Type': 'application/json',
            'Connection': 'keep-alive',
            ...fetchOptions?.headers,
          },
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const error = await response.text();
          throw new Error(`API error: ${response.status} - ${error}`);
        }

        return response.json();
      } catch (error) {
        clearTimeout(timeoutId);
        lastError = error instanceof Error ? error : new Error(String(error));

        // Don't retry on 4xx errors (client errors)
        if (lastError.message.includes('API error: 4')) {
          throw lastError;
        }

        // Retry with exponential backoff
        if (attempt < retries) {
          const delay = getRetryDelay(attempt);
          console.log(`Request failed, retrying in ${delay}ms...`);
          await sleep(delay);
        }
      }
    }

    throw lastError || new Error('Request failed after retries');
  };

  // Store promise for deduplication
  const promise = executeRequest().finally(() => {
    pendingRequests.delete(cacheKey);
  });

  if (dedupe) {
    pendingRequests.set(cacheKey, promise);
  }

  return promise;
}
```

**Key Features:**
- **AbortController**: Implements request timeout
- **Exponential Backoff**: 500ms → 1s → 2s delays
- **Deduplication**: Same GET requests share one in-flight promise
- **Keep-Alive**: Reuses HTTP connections

### 23.2 API Endpoints

```typescript
// Health check with fast timeout
export async function healthCheck(): Promise<{ status: string }> {
  return fetchWithRetry(`${API_BASE}/health`, { timeout: FAST_TIMEOUT, retries: 2 });
}

// Log queries with longer timeout
export async function queryLogs(params: {
  from: string;
  to: string;
  service_name?: string;
  severity_min?: number;
  limit?: number;
  offset?: number;
}): Promise<LogQueryResponse> {
  const searchParams = new URLSearchParams();
  searchParams.set('from', params.from);
  searchParams.set('to', params.to);
  
  if (params.service_name) searchParams.set('service_name', params.service_name);
  if (params.severity_min !== undefined) searchParams.set('severity_min', params.severity_min.toString());
  if (params.limit) searchParams.set('limit', params.limit.toString());
  if (params.offset) searchParams.set('offset', params.offset.toString());

  return fetchWithRetry(`${API_BASE}/logs?${searchParams}`, { timeout: LOG_QUERY_TIMEOUT });
}

// ML Chat with extended timeout for LLM
export async function mlChat(params: {
  message: string;
  mode?: 'auto' | 'anomaly' | 'classify' | 'security';
  fast_mode?: boolean;
  use_llm?: boolean;
}, signal?: AbortSignal): Promise<MLChatResponse> {
  const useLLM = params.use_llm ?? !params.fast_mode;
  
  // Use dedicated API route for LLM to bypass proxy timeout
  const endpoint = useLLM ? '/api/ml/chat' : `${API_BASE}/ml/chat`;
  const timeout = useLLM ? 400000 : 15000;  // 400s LLM, 15s fast mode
  
  return fetchWithRetry(endpoint, {
    method: 'POST',
    body: JSON.stringify({
      message: params.message,
      mode: params.mode || 'auto',
      fast_mode: params.fast_mode ?? true,
      use_llm: useLLM,
    }),
    timeout,
    retries: useLLM ? 0 : MAX_RETRIES,  // No retries for LLM
    dedupe: false,  // Don't dedupe POST requests
    signal,
  });
}
```

### 23.3 Time Range Helper

```typescript
export function getTimeRange(range: string): { from: string; to: string } {
  const now = new Date();
  const to = now.toISOString();
  let from: Date;

  switch (range) {
    case '1h':
      from = new Date(now.getTime() - 60 * 60 * 1000);
      break;
    case '6h':
      from = new Date(now.getTime() - 6 * 60 * 60 * 1000);
      break;
    case '24h':
      from = new Date(now.getTime() - 24 * 60 * 60 * 1000);
      break;
    case '7d':
      from = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      break;
    case '30d':
      from = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
      break;
    case 'all':
      from = new Date('2000-01-01');  // Effectively "all time"
      break;
    default:
      from = new Date(now.getTime() - 24 * 60 * 60 * 1000);
  }

  return { from: from.toISOString(), to };
}
```

---

## 24. AIChat Component (AIChat.tsx)

**File:** `apps/ui/src/components/AIChat.tsx`

### Purpose
React component for AI-powered chat interface with:
- Fast Mode / LLM Mode toggle
- Request cancellation
- Sample log analysis
- Model status display

### Component State

```tsx
export default function AIChat() {
  // Message history
  const [messages, setMessages] = useState<Message[]>([]);
  // Input field value
  const [input, setInput] = useState('');
  // Loading state
  const [loading, setLoading] = useState(false);
  // Error display
  const [error, setError] = useState<string | null>(null);
  // ML model status
  const [modelStatus, setModelStatus] = useState<ModelTrainingStatus | null>(null);
  // Fast mode toggle (default: true for instant responses)
  const [fastMode, setFastMode] = useState(true);
  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
```

### 24.1 Message Submission

```tsx
const handleSubmit = async (e?: React.FormEvent) => {
  e?.preventDefault();
  if (!input.trim() || loading) return;
  await sendMessage(input.trim());
};

const sendMessage = async (message: string) => {
  setInput('');
  setError(null);

  // Add user message to history
  const userMessage: Message = {
    id: Date.now().toString(),
    role: 'user',
    content: message,
    timestamp: new Date(),
  };
  setMessages((prev) => [...prev, userMessage]);

  setLoading(true);
  
  // Create abort controller for cancellation
  abortControllerRef.current = new AbortController();

  try {
    const response = await mlChat({ 
      message, 
      mode: 'auto',
      fast_mode: fastMode,
      use_llm: !fastMode,
    }, abortControllerRef.current.signal);
    
    // Check if request was cancelled
    if (abortControllerRef.current?.signal.aborted) {
      return;
    }

    // Add assistant response
    const assistantMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: response.response,
      analysis: response.analysis,
      modelStatus: response.model_status,
      suggestions: response.suggestions,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, assistantMessage]);
    
  } catch (e) {
    if (e instanceof Error && e.name === 'AbortError') {
      // Request was cancelled, don't show error
      return;
    }
    setError(e instanceof Error ? e.message : 'Failed to get response');
  } finally {
    setLoading(false);
    abortControllerRef.current = null;
  }
};
```

### 24.2 Request Cancellation

```tsx
const handleCancel = () => {
  if (abortControllerRef.current) {
    abortControllerRef.current.abort();
    setLoading(false);
    
    // Add cancelled message
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        role: 'assistant',
        content: '⚠️ Request cancelled by user.',
        timestamp: new Date(),
      },
    ]);
  }
};
```

### 24.3 Mode Toggle UI

```tsx
{/* Fast Mode / LLM Mode Toggle */}
<div className="flex items-center gap-2 text-sm">
  <button
    onClick={() => setFastMode(!fastMode)}
    className={`flex items-center gap-1 px-3 py-1.5 rounded-lg transition-all ${
      fastMode
        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
        : 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
    }`}
  >
    {fastMode ? (
      <>
        <Zap className="w-4 h-4" />
        <span>Fast Mode</span>
      </>
    ) : (
      <>
        <Bot className="w-4 h-4" />
        <span>LLM Mode</span>
      </>
    )}
  </button>
  <span className="text-slate-500 text-xs">
    {fastMode 
      ? 'Instant responses using built-in knowledge' 
      : 'AI-powered responses (slower)'}
  </span>
</div>
```

### 24.4 Input Field with Cancel Button

```tsx
<form onSubmit={handleSubmit} className="flex gap-2">
  <input
    ref={inputRef}
    type="text"
    value={input}
    onChange={(e) => setInput(e.target.value)}
    placeholder="Ask about logs, security, or paste a log to analyze..."
    className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 
               text-slate-900 placeholder-slate-400
               focus:ring-2 focus:ring-blue-500 focus:border-transparent"
    disabled={loading}
  />
  
  {loading ? (
    // Cancel button during loading
    <button
      type="button"
      onClick={handleCancel}
      className="px-4 py-2 bg-gradient-to-r from-red-600 to-red-700 
                 text-white rounded-lg flex items-center gap-2 
                 hover:from-red-700 hover:to-red-800"
    >
      <Square className="w-4 h-4 fill-current" />
      Cancel
    </button>
  ) : (
    // Send button
    <button
      type="submit"
      disabled={!input.trim()}
      className="px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 
                 text-white rounded-lg flex items-center gap-2 
                 disabled:opacity-50 hover:from-blue-700 hover:to-purple-700"
    >
      <Send className="w-4 h-4" />
      Send
    </button>
  )}
</form>
```

### 24.5 Message Rendering with Markdown

```tsx
{messages.map((message) => (
  <div
    key={message.id}
    className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
  >
    {/* Avatar */}
    <div className={`w-8 h-8 rounded-full flex items-center justify-center
      ${message.role === 'user' 
        ? 'bg-blue-500/20' 
        : 'bg-gradient-to-br from-purple-500/20 to-pink-500/20'}`}
    >
      {message.role === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
    </div>
    
    {/* Message Content */}
    <div className={`max-w-[80%] rounded-lg p-4 ${
      message.role === 'user' 
        ? 'bg-blue-500/10 border border-blue-500/20' 
        : 'bg-slate-800 border border-slate-700'
    }`}>
      {/* Render markdown content */}
      <ReactMarkdown 
        className="prose prose-invert prose-sm max-w-none"
        components={{
          // Custom code block rendering
          code: ({ node, inline, className, children, ...props }) => (
            inline ? (
              <code className="bg-slate-700 px-1 rounded text-sm">{children}</code>
            ) : (
              <pre className="bg-slate-900 p-3 rounded overflow-x-auto">
                <code>{children}</code>
              </pre>
            )
          ),
          // Custom table rendering
          table: ({ children }) => (
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">{children}</table>
            </div>
          ),
        }}
      >
        {message.content}
      </ReactMarkdown>
    </div>
  </div>
))}
```

---

## 25. LogExplorer Component (LogExplorer.tsx)

**File:** `apps/ui/src/components/LogExplorer.tsx`

### Purpose
Displays and filters log entries with:
- Time range selection
- Service filtering
- Severity filtering
- Template-based grouping
- Expandable log details

### Component Props and State

```tsx
interface Props {
  serviceName: string;
  timeRange: string;
  refreshTrigger?: number;
}

export default function LogExplorer({ serviceName, timeRange, refreshTrigger }: Props) {
  const [logs, setLogs] = useState<LogEvent[]>([]);
  const [total, setTotal] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [severityFilter, setSeverityFilter] = useState<number | undefined>(undefined);
  const [expandedLogs, setExpandedLogs] = useState<Set<number>>(new Set());
  const [offset, setOffset] = useState(0);
  const [showRawLogs, setShowRawLogs] = useState(true);
  const limit = 100;
```

### 25.1 Log Loading with Filters

```tsx
const loadLogs = useCallback(async () => {
  setLoading(true);
  setError(null);

  try {
    const { from, to } = getTimeRange(timeRange);
    
    const response = await queryLogs({
      from,
      to,
      service_name: serviceName || undefined,
      severity_min: severityFilter,
      limit,
      offset,
    });

    setLogs(response.logs);
    setTotal(response.total);
    setHasMore(response.has_more);
  } catch (e) {
    setError(e instanceof Error ? e.message : 'Failed to load logs');
  } finally {
    setLoading(false);
  }
}, [serviceName, timeRange, severityFilter, offset]);

// Reload when filters change
useEffect(() => {
  loadLogs();
}, [loadLogs, refreshTrigger]);

// Reset offset when filters change
useEffect(() => {
  setOffset(0);
}, [serviceName, timeRange, severityFilter]);
```

### 25.2 Severity Styling

```typescript
// Severity level colors (in types.ts)
export const SEVERITY_COLORS: Record<number, string> = {
  0: 'text-slate-400',    // DEBUG
  1: 'text-blue-400',     // INFO
  2: 'text-yellow-400',   // WARNING
  3: 'text-orange-400',   // ERROR
  4: 'text-red-400',      // CRITICAL
};

export const SEVERITY_BG_COLORS: Record<number, string> = {
  0: 'bg-slate-400/10',
  1: 'bg-blue-400/10',
  2: 'bg-yellow-400/10',
  3: 'bg-orange-400/10',
  4: 'bg-red-400/10',
};

export const SEVERITY_NAMES: Record<number, string> = {
  0: 'DEBUG',
  1: 'INFO',
  2: 'WARNING',
  3: 'ERROR',
  4: 'CRITICAL',
};
```

### 25.3 Filter UI

```tsx
{/* Severity Filter */}
<div className="flex gap-2">
  <select
    value={severityFilter ?? ''}
    onChange={(e) => setSeverityFilter(e.target.value ? parseInt(e.target.value) : undefined)}
    className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm"
  >
    <option value="">All Severities</option>
    <option value="0">DEBUG & above</option>
    <option value="1">INFO & above</option>
    <option value="2">WARNING & above</option>
    <option value="3">ERROR & above</option>
    <option value="4">CRITICAL only</option>
  </select>
</div>

{/* Search within results */}
<div className="relative flex-1">
  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
  <input
    type="text"
    placeholder="Search in logs..."
    value={searchQuery}
    onChange={(e) => setSearchQuery(e.target.value)}
    className="w-full bg-slate-800 border border-slate-700 rounded-lg pl-10 pr-4 py-2 text-sm"
  />
</div>
```

### 25.4 Log Entry Display

```tsx
{filteredLogs.map((log) => (
  <div
    key={log.id}
    className={`border-b border-slate-800 hover:bg-slate-800/50 transition-colors
      ${SEVERITY_BG_COLORS[log.severity]}`}
  >
    {/* Log header row */}
    <div
      className="flex items-center gap-4 px-4 py-3 cursor-pointer"
      onClick={() => toggleExpand(log.id)}
    >
      {/* Expand/collapse icon */}
      {expandedLogs.has(log.id) ? (
        <ChevronDown className="w-4 h-4 text-slate-500" />
      ) : (
        <ChevronRight className="w-4 h-4 text-slate-500" />
      )}
      
      {/* Timestamp */}
      <span className="text-slate-500 text-sm w-40 flex-shrink-0">
        {formatTimestamp(log.timestamp_utc)}
      </span>
      
      {/* Service */}
      <span className="text-slate-400 text-sm w-24 flex-shrink-0">
        {log.service_name}
      </span>
      
      {/* Severity badge */}
      <span className={`text-xs px-2 py-0.5 rounded w-20 text-center
        ${SEVERITY_COLORS[log.severity]} ${SEVERITY_BG_COLORS[log.severity]}`}>
        {SEVERITY_NAMES[log.severity]}
      </span>
      
      {/* Log message (truncated) */}
      <span className="text-slate-300 text-sm flex-1 truncate">
        {showRawLogs ? log.body_raw : log.template_text}
      </span>
    </div>
    
    {/* Expanded details */}
    {expandedLogs.has(log.id) && (
      <div className="px-4 pb-4 pt-2 bg-slate-900/50 border-t border-slate-800">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-slate-500">Template:</span>
            <code className="ml-2 text-purple-400">{log.template_text}</code>
          </div>
          <div>
            <span className="text-slate-500">Parameters:</span>
            <code className="ml-2 text-emerald-400">
              {JSON.stringify(log.parameters)}
            </code>
          </div>
          {log.trace_id && (
            <div>
              <span className="text-slate-500">Trace ID:</span>
              <code className="ml-2 text-blue-400">{log.trace_id}</code>
            </div>
          )}
        </div>
        <div className="mt-3">
          <span className="text-slate-500 text-sm">Full Message:</span>
          <pre className="mt-1 p-3 bg-slate-950 rounded text-sm text-slate-300 overflow-x-auto">
            {log.body_raw}
          </pre>
        </div>
      </div>
    )}
  </div>
))}
```

### 25.5 Pagination

```tsx
{/* Pagination controls */}
<div className="flex items-center justify-between px-4 py-3 border-t border-slate-800">
  <span className="text-sm text-slate-500">
    Showing {offset + 1} - {offset + logs.length} of {total}
  </span>
  
  <div className="flex gap-2">
    <button
      onClick={() => setOffset(Math.max(0, offset - limit))}
      disabled={offset === 0}
      className="px-3 py-1.5 bg-slate-800 rounded text-sm disabled:opacity-50"
    >
      Previous
    </button>
    <button
      onClick={() => setOffset(offset + limit)}
      disabled={!hasMore}
      className="px-3 py-1.5 bg-slate-800 rounded text-sm disabled:opacity-50"
    >
      Next
    </button>
  </div>
</div>
```

---

# Appendix: Key Patterns Used

## A. Repository Pattern
All database access goes through repository classes (`LogsRepo`, `TemplatesRepo`) that encapsulate SQL queries.

## B. Service Pattern
Business logic is in service classes (`IngestService`, `ChatService`) that coordinate between repositories and external systems.

## C. Dependency Injection
Global instances are lazily created and accessed via getter functions (`get_settings()`, `get_ollama_client()`).

## D. Async/Await Throughout
All I/O operations use Python's `async/await` for non-blocking execution.

## E. Batch Processing
Large operations use batching (5000 events at a time) to balance memory usage and performance.

## F. Connection Pooling
Database connections are pooled to reduce connection overhead.

## G. Request Deduplication
The frontend deduplicates identical concurrent requests to reduce server load.

---

*End of Code Documentation*

*Document Version: 1.0.0*
*Last Updated: December 29, 2025*





