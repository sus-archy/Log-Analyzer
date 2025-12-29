"""
LogMind AI - FastAPI Application

Local log observability platform with template mining and semantic search.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Core imports
from .core.config import settings
from .core.logging import setup_logging, get_logger
from .core.rate_limit import limiter

# Storage imports
from .storage.db import init_db, close_db

# Service imports
from .vector.faiss_index import init_faiss_index
from .services.embedding_service import get_embedding_service
from .llm.ollama_client import close_ollama_client

# Route imports (all from routes package)
from .routes import (
    health_router,
    auth_router,
    ingest_router,
    logs_router,
    templates_router,
    semantic_router,
    chat_router,
    metrics_router,
    ml_router,
)

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting LogMind AI...")
    logger.info(f"Auth enabled: {settings.auth_enabled}")
    logger.info(f"Rate limiting enabled: {settings.rate_limit_enabled}")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Initialize FAISS index
    await init_faiss_index()
    
    # Try to load/rebuild embeddings
    embedding_service = get_embedding_service()
    await embedding_service.ensure_index_loaded()
    
    # Initialize ML models
    try:
        from .ml import initialize_models
        ml_status = await initialize_models()
        logger.info(f"ML models initialized: {ml_status}")
    except Exception as e:
        logger.warning(f"ML model initialization failed (non-critical): {e}")
    
    logger.info("LogMind AI started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LogMind AI...")
    
    # Clear cache on shutdown
    from .core.cache import get_cache
    await get_cache().clear()
    
    await close_db()
    await close_ollama_client()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="LogMind AI",
    description="Local log observability platform with template mining and semantic search",
    version="1.0.0",
    lifespan=lifespan,
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Parse CORS origins from settings
cors_origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]

# Add CORS middleware with secure settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After"],
    max_age=3600,  # Cache CORS preflight for 1 hour
)

# Include routers - organized by category
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


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        timeout_keep_alive=65,  # Keep connections alive longer
        limit_concurrency=100,   # Handle more concurrent requests
    )

