"""Routes package."""

from .health import router as health_router
from .ingest import router as ingest_router
from .logs import router as logs_router
from .templates import router as templates_router
from .semantic_search import router as semantic_router
from .chat import router as chat_router

__all__ = [
    "health_router",
    "ingest_router",
    "logs_router",
    "templates_router",
    "semantic_router",
    "chat_router",
]
