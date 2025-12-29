"""
Rate limiting module using SlowAPI.

Provides request rate limiting to prevent abuse of expensive endpoints.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse

from .config import settings
from .logging import get_logger

logger = get_logger(__name__)


def get_user_identifier(request: Request) -> str:
    """
    Get identifier for rate limiting.
    
    Uses authenticated username if available, otherwise IP address.
    """
    # Try to get user from request state (set by auth middleware)
    user = getattr(request.state, "user", None)
    if user and hasattr(user, "username"):
        return f"user:{user.username}"
    
    # Fall back to IP address
    return get_remote_address(request)


# Create limiter instance
limiter = Limiter(
    key_func=get_user_identifier,
    default_limits=[settings.rate_limit_default],
    storage_uri="memory://",  # Use Redis in production: "redis://localhost:6379"
    enabled=settings.rate_limit_enabled,
)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Handle rate limit exceeded errors."""
    logger.warning(f"Rate limit exceeded for {get_user_identifier(request)}: {exc.detail}")
    
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": str(exc.detail),
            "retry_after": getattr(exc, "retry_after", 60),
        },
        headers={
            "Retry-After": str(getattr(exc, "retry_after", 60)),
            "X-RateLimit-Limit": str(getattr(exc, "limit", "unknown")),
        }
    )


# Rate limit decorators for specific endpoints
# Usage: @limiter.limit("10/minute")
RATE_LIMITS = {
    "chat": settings.rate_limit_chat,           # e.g., "10/minute"
    "embed": settings.rate_limit_embed,         # e.g., "5/minute"
    "ingest": settings.rate_limit_ingest,       # e.g., "20/minute"
    "search": settings.rate_limit_search,       # e.g., "30/minute"
    "default": settings.rate_limit_default,     # e.g., "100/minute"
}
