"""
LogMind AI - FastAPI Application

Local log observability platform with template mining and semantic search.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.logging import setup_logging, get_logger
from .storage.db import init_db, close_db
from .vector.faiss_index import init_faiss_index
from .services.embedding_service import get_embedding_service
from .llm.ollama_client import close_ollama_client
from .routes import (
    health_router,
    ingest_router,
    logs_router,
    templates_router,
    semantic_router,
    chat_router,
)
from .routes.metrics import router as metrics_router

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting LogMind AI...")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Initialize FAISS index
    await init_faiss_index()
    
    # Try to load/rebuild embeddings
    embedding_service = get_embedding_service()
    await embedding_service.ensure_index_loaded()
    
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

# Add CORS middleware with optimized settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],  # Expose custom headers
    max_age=3600,  # Cache CORS preflight for 1 hour
)

# Include routers
app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(logs_router)
app.include_router(templates_router)
app.include_router(semantic_router)
app.include_router(chat_router)
app.include_router(metrics_router)


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

