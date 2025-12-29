"""
Routes package.

API endpoint routers for LogMind AI.
"""

from .health import router as health_router
from .ingest import router as ingest_router
from .logs import router as logs_router
from .templates import router as templates_router
from .semantic_search import router as semantic_router
from .chat import router as chat_router
from .metrics import router as metrics_router
from .auth import router as auth_router
from .ml import router as ml_router

__all__ = [
    # Core routers
    "health_router",
    "auth_router",
    # Data routers
    "ingest_router",
    "logs_router",
    "templates_router",
    # AI/Search routers
    "semantic_router",
    "chat_router",
    "ml_router",
    # Monitoring routers
    "metrics_router",
]
