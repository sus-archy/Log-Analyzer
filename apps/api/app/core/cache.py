"""
In-memory caching for frequently accessed data.

Provides fast response times for common queries.
"""

import asyncio
import time
from typing import Any, Callable, Optional, Dict
from functools import wraps
from dataclasses import dataclass, field

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with expiration."""
    value: Any
    expires_at: float
    created_at: float = field(default_factory=time.time)


class MemoryCache:
    """
    Simple in-memory cache with TTL support.
    
    Features:
    - Time-based expiration
    - Automatic cleanup of expired entries
    - Thread-safe with asyncio locks
    - Maximum size limit to prevent memory issues
    """
    
    def __init__(self, max_size: int = 1000, cleanup_interval: float = 60.0):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._max_size = max_size
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        self._hits = 0
        self._misses = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        async with self._lock:
            await self._maybe_cleanup()
            
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            
            if time.time() > entry.expires_at:
                del self._cache[key]
                self._misses += 1
                return None
            
            self._hits += 1
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: float = 60.0) -> None:
        """Set value in cache with TTL in seconds."""
        async with self._lock:
            # Enforce max size by removing oldest entries
            if len(self._cache) >= self._max_size:
                await self._evict_oldest()
            
            self._cache[key] = CacheEntry(
                value=value,
                expires_at=time.time() + ttl
            )
    
    async def delete(self, key: str) -> bool:
        """Delete a specific key from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern (simple prefix match)."""
        async with self._lock:
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(pattern)]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)
    
    async def _maybe_cleanup(self) -> None:
        """Periodically clean up expired entries."""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        self._last_cleanup = current_time
        expired_keys = [
            k for k, v in self._cache.items() 
            if current_time > v.expires_at
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def _evict_oldest(self) -> None:
        """Remove oldest entries when cache is full."""
        if not self._cache:
            return
        
        # Sort by creation time and remove oldest 10%
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at
        )
        
        to_remove = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:to_remove]:
            del self._cache[key]
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }


# Global cache instance
_cache = MemoryCache(max_size=500, cleanup_interval=60.0)


def get_cache() -> MemoryCache:
    """Get the global cache instance."""
    return _cache


def cached(ttl: float = 60.0, key_prefix: str = ""):
    """
    Decorator for caching async function results.
    
    Usage:
        @cached(ttl=300, key_prefix="stats")
        async def get_expensive_stats(service: str):
            ...
    
    Args:
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache keys to allow selective invalidation
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key from function name and arguments
            cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            
            cache = get_cache()
            cached_value = await cache.get(cache_key)
            
            if cached_value is not None:
                return cached_value
            
            # Call the actual function
            result = await func(*args, **kwargs)
            
            # Cache the result
            await cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


# TTL constants for different types of data
class CacheTTL:
    """Standard TTL values for different data types."""
    QUICK_STATS = 30.0      # Quick stats refresh every 30 seconds
    SERVICES_LIST = 60.0     # Services list refresh every minute
    TEMPLATES = 120.0        # Templates refresh every 2 minutes
    METRICS = 60.0           # Metrics refresh every minute
    HEALTH = 10.0            # Health checks every 10 seconds
    EMBEDDING_STATS = 30.0   # Embedding stats every 30 seconds
