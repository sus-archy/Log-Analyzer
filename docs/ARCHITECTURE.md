# LogMind AI - Architecture Documentation

## Overview

LogMind AI is a local-first log observability platform with AI-powered insights. This document describes the architectural decisions and implementation details.

## Version 2.0 Changes

This version introduces significant improvements to security, performance, and maintainability:

### Security Improvements

#### 1. JWT Authentication (`/app/core/security.py`)

Optional JWT-based authentication with:
- OAuth2 password flow
- Configurable via `AUTH_ENABLED` environment variable
- Secure password hashing with bcrypt
- Token expiration (configurable)

```python
# Usage
from app.core.security import get_current_user

@router.get("/protected")
async def protected_route(user: dict = Depends(get_current_user)):
    return {"user": user}
```

#### 2. CORS Configuration

Properly configured CORS with:
- Configurable origins via `CORS_ORIGINS` environment variable
- JSON array format: `["http://localhost:3000", "https://myapp.com"]`
- Credentials support enabled

#### 3. Rate Limiting (`/app/core/rate_limit.py`)

SlowAPI-based rate limiting:
- Per-endpoint configurable limits
- User identification by IP (or user ID when authenticated)
- Graceful 429 responses with retry-after headers

```python
# Configuration
RATE_LIMITS = {
    "chat": "20/minute",
    "ingest": "5/minute",
    "semantic_search": "30/minute",
}
```

#### 4. Input Validation (`/app/schemas/chat.py`)

Enhanced Pydantic validation:
- Message length limits (1-10000 characters)
- HTML/script tag sanitization
- Whitespace normalization
- Valid value bounds for all parameters

### Performance Improvements

#### 1. Database Connection Pool (`/app/storage/db.py`)

Replaced single connection with connection pool:
- Configurable pool size (default: 5)
- Acquire/release pattern with automatic cleanup
- Timeout handling for exhausted pools
- Health check and statistics

```python
# Usage
db = await get_db()
try:
    result = await db.execute("SELECT ...")
finally:
    await release_db(db)
```

#### 2. Database Indexes (`/app/storage/migrations.py`)

New indexes for common query patterns:
- `idx_logs_timestamp` - Time-range queries
- `idx_logs_severity` - Severity filtering
- `idx_logs_tenant_time` - Composite tenant + time
- `idx_logs_service` - Service filtering

Automatic ANALYZE for query optimizer.

#### 3. Optimized Security Metrics

Replaced `ORDER BY RANDOM()` sampling with stratified sampling:
- Sample high-severity logs first
- Sample proportionally from each service
- Deterministic, reproducible results

### Code Quality Improvements

#### 1. Service Separation

Split `MetricsService` into:
- `MetricsService` - Performance metrics
- `SecurityMetricsService` - Security analysis

Benefits:
- Single responsibility
- Easier testing
- Independent scaling

#### 2. Configurable Security Patterns (`/app/services/security_patterns.json`)

Security pattern detection moved to JSON configuration:
- Easily editable without code changes
- Category-based patterns with severity boosts
- Configurable thresholds and scoring

```json
{
  "patterns": {
    "authentication_failures": {
      "keywords": ["failed password", "login fail", ...],
      "severity_boost": 1.5
    }
  }
}
```

#### 3. Enhanced Error Handling

- `OllamaError` now includes `status_code` and `response_body`
- All services properly release database connections in `finally` blocks
- Consistent error response format

### Frontend Improvements

#### 1. Error Boundaries (`/components/ErrorBoundary.tsx`)

React error boundaries for graceful failure:
- Catches JavaScript errors in component tree
- Shows user-friendly error UI
- Development mode shows stack traces
- Retry functionality

```tsx
<ErrorBoundary>
  <YourComponent />
</ErrorBoundary>
```

#### 2. Zustand State Management (`/lib/store.ts`)

Centralized state management:
- Connection status (API + Ollama)
- Global filters
- Error queue
- Persisted preferences (dark mode, sidebar)

```tsx
const connection = useConnection();
const { checkConnections } = useConnectionActions();
```

#### 3. Connection Status Component (`/components/ConnectionStatus.tsx`)

Real-time connection monitoring:
- Shows API and Ollama status
- Automatic periodic health checks
- Actionable error messages

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_ENABLED` | `false` | Enable JWT authentication |
| `SECRET_KEY` | (random) | JWT signing key (generate for production) |
| `ALGORITHM` | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Token lifetime |
| `ADMIN_USERNAME` | `admin` | Default admin username |
| `ADMIN_PASSWORD_HASH` | (hash of 'changeme') | Admin password hash |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed CORS origins |
| `RATE_LIMIT_CHAT` | `20/minute` | Chat endpoint rate limit |
| `RATE_LIMIT_INGEST` | `5/minute` | Ingest endpoint rate limit |
| `DB_POOL_SIZE` | `5` | Database connection pool size |
| `DB_POOL_TIMEOUT` | `30.0` | Pool acquire timeout (seconds) |

## Testing

New test files:

- `test_security_metrics.py` - Security pattern detection tests
- `test_db_pool.py` - Connection pool tests
- `test_validation.py` - Input validation tests

Run tests:
```bash
cd apps/api
pytest tests/ -v
```

## Migration Guide

### From v1 to v2

1. **Install new dependencies:**
   ```bash
   pip install -r apps/api/requirements.txt
   cd apps/ui && npm install
   ```

2. **Run database migration:**
   Migration runs automatically on startup. To run manually:
   ```python
   from app.storage.migrations import run_migrations
   await run_migrations()
   ```

3. **Update configuration:**
   - Set `CORS_ORIGINS` if using non-localhost frontend
   - Set `SECRET_KEY` for production authentication
   - Adjust rate limits as needed

4. **Enable authentication (optional):**
   ```bash
   export AUTH_ENABLED=true
   export ADMIN_PASSWORD_HASH=$(python -c "from passlib.context import CryptContext; print(CryptContext(schemes=['bcrypt']).hash('yourpassword'))")
   ```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Next.js UI                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Zustand Store                                           │  │
│  │  ├─ Connection Status (API/Ollama health)               │  │
│  │  ├─ Global Filters                                       │  │
│  │  └─ Error Queue                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Log Explorer│  │  Templates  │  │      Chat Panel         │ │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘ │
│         │                │                      │               │
│  ┌──────▼────────────────▼──────────────────────▼─────────────┐│
│  │              ErrorBoundary + ConnectionStatus              ││
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP (with retry + dedup)
┌────────────────────────────▼────────────────────────────────────┐
│                      FastAPI Backend                            │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Middleware Stack                                        │  │
│  │  ├─ CORS (configurable origins)                          │  │
│  │  ├─ Rate Limiting (SlowAPI)                              │  │
│  │  └─ Request Logging                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Auth Layer (Optional)                                   │  │
│  │  ├─ JWT Token Validation                                 │  │
│  │  └─ OAuth2 Password Flow                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌────────────────────┐ │
│  │ Ingest  │ │  Logs    │ │ Templates │ │ Semantic + Chat    │ │
│  │ Service │ │  Service │ │  Service  │ │ Services           │ │
│  └────┬────┘ └────┬─────┘ └─────┬─────┘ └──────────┬─────────┘ │
│       │           │             │                   │           │
│  ┌────▼──────┐ ┌──▼──────┐                          │           │
│  │ Metrics   │ │ Security│                          │           │
│  │ Service   │ │ Metrics │                          │           │
│  └────┬──────┘ └────┬────┘                          │           │
│       │             │                               │           │
│  ┌────▼─────────────▼───────────────────────────────▼────────┐ │
│  │                     Storage Layer                         │ │
│  │  ┌───────────────────────────────────────────────────────┐│ │
│  │  │              Connection Pool (5 connections)          ││ │
│  │  │  ├─ Acquire/Release pattern                           ││ │
│  │  │  ├─ Timeout handling                                  ││ │
│  │  │  └─ Health monitoring                                 ││ │
│  │  └───────────────────────────────────────────────────────┘│ │
│  │  ┌───────────────────────────────────────────────────────┐│ │
│  │  │                 SQLite Database (WAL mode)            ││ │
│  │  │  • Optimized indexes for queries                      ││ │
│  │  │  • Automatic migrations                               ││ │
│  │  └───────────────────────────────────────────────────────┘│ │
│  │  ┌───────────────────────────────────────────────────────┐│ │
│  │  │                 FAISS Index                           ││ │
│  │  │  • Vector similarity search (IndexFlatIP)             ││ │
│  │  └───────────────────────────────────────────────────────┘│ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                        Ollama (Local)                           │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐ │
│  │  qwen2.5:3b         │  │  nomic-embed-text                │ │
│  │  (Chat Model)       │  │  (Embedding Model)               │ │
│  └─────────────────────┘  └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Future Improvements

See `features_roadmap.txt` for planned features.
