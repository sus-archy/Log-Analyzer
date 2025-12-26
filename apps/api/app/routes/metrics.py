"""
Metrics API routes - performance and security metrics endpoints.
Optimized with caching for fast responses.
"""

from typing import Optional
from fastapi import APIRouter, Query, HTTPException

from ..core.logging import get_logger
from ..core.cache import get_cache, CacheTTL
from ..services.metrics_service import get_metrics_service

logger = get_logger(__name__)
router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/performance")
async def get_performance_metrics(
    service_name: Optional[str] = Query(None, description="Filter by service"),
    from_time: Optional[str] = Query(None, alias="from", description="Start time (ISO 8601)"),
    to_time: Optional[str] = Query(None, alias="to", description="End time (ISO 8601)"),
):
    """Get performance metrics with caching."""
    cache = get_cache()
    cache_key = f"perf_metrics:{service_name or 'all'}:{from_time}:{to_time}"
    
    # Try cache first
    cached = await cache.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        service = get_metrics_service()
        result = await service.get_performance_metrics(
            service_name=service_name,
            from_time=from_time,
            to_time=to_time,
        )
        await cache.set(cache_key, result, CacheTTL.METRICS)
        return result
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/security")
async def get_security_metrics(
    service_name: Optional[str] = Query(None, description="Filter by service"),
    from_time: Optional[str] = Query(None, alias="from", description="Start time (ISO 8601)"),
    to_time: Optional[str] = Query(None, alias="to", description="End time (ISO 8601)"),
):
    """Get security metrics with caching."""
    cache = get_cache()
    cache_key = f"sec_metrics:{service_name or 'all'}:{from_time}:{to_time}"
    
    # Try cache first
    cached = await cache.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        service = get_metrics_service()
        result = await service.get_security_metrics(
            service_name=service_name,
            from_time=from_time,
            to_time=to_time,
        )
        await cache.set(cache_key, result, CacheTTL.METRICS)
        return result
    except Exception as e:
        logger.error(f"Security metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services/health")
async def get_services_health(
    from_time: Optional[str] = Query(None, alias="from", description="Start time (ISO 8601)"),
    to_time: Optional[str] = Query(None, alias="to", description="End time (ISO 8601)"),
):
    """Get services health with caching."""
    cache = get_cache()
    cache_key = f"svc_health:{from_time}:{to_time}"
    
    # Try cache first
    cached = await cache.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        service = get_metrics_service()
        result = await service.get_service_health(
            from_time=from_time,
            to_time=to_time,
        )
        await cache.set(cache_key, result, CacheTTL.METRICS)
        return result
    except Exception as e:
        logger.error(f"Services health error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report")
async def get_full_report(
    service_name: Optional[str] = Query(None, description="Filter by service"),
    from_time: Optional[str] = Query(None, alias="from", description="Start time (ISO 8601)"),
    to_time: Optional[str] = Query(None, alias="to", description="End time (ISO 8601)"),
):
    """Get full report with caching."""
    from datetime import datetime
    
    cache = get_cache()
    cache_key = f"full_report:{service_name or 'all'}:{from_time}:{to_time}"
    
    # Try cache first
    cached = await cache.get(cache_key)
    if cached is not None:
        return cached
    
    try:
        service = get_metrics_service()
        
        performance = await service.get_performance_metrics(
            service_name=service_name,
            from_time=from_time,
            to_time=to_time,
        )
        
        security = await service.get_security_metrics(
            service_name=service_name,
            from_time=from_time,
            to_time=to_time,
        )
        
        services_health = await service.get_service_health(
            from_time=from_time,
            to_time=to_time,
        )
        
        result = {
            "generated_at": datetime.utcnow().isoformat(),
            "filters": {
                "service_name": service_name,
                "from_time": from_time,
                "to_time": to_time,
            },
            "performance": performance,
            "security": security,
            "services_health": services_health,
        }
        
        await cache.set(cache_key, result, CacheTTL.METRICS)
        return result
    except Exception as e:
        logger.error(f"Full report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
