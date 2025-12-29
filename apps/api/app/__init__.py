"""
LogMind AI - API Application

A local-first log observability platform with AI-powered insights.

Modules:
    - core: Configuration, logging, security, caching
    - storage: Database and repository layer
    - services: Business logic services
    - routes: API endpoints
    - ml: Machine learning models and training
    - llm: LLM integration (Ollama)
    - vector: FAISS vector search
    - parsers: Log parsing utilities
    - schemas: Pydantic request/response models
"""

from .main import app

__version__ = "0.2.0"
__all__ = ["app", "__version__"]
