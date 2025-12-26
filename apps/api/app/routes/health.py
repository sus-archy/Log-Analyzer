"""
Health check routes.
"""

from fastapi import APIRouter

from ..core.cache import get_cache

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@router.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "LogMind AI",
        "version": "1.0.0",
        "status": "running",
    }


@router.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    cache = get_cache()
    return cache.stats()


@router.post("/cache/clear")
async def clear_cache():
    """Clear the cache."""
    cache = get_cache()
    await cache.clear()
    return {"status": "cleared"}
