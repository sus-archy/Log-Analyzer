"""
Semantic search routes.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..core.logging import get_logger
from ..schemas.chat import SemanticSearchResponse
from ..services.semantic_service import get_semantic_service

logger = get_logger(__name__)
router = APIRouter(prefix="/semantic_search", tags=["semantic"])


@router.get("", response_model=SemanticSearchResponse)
async def semantic_search(
    q: str = Query(..., min_length=1, description="Search query"),
    tenant_id: Optional[str] = Query(None, description="Tenant ID"),
    service_name: Optional[str] = Query("", description="Service name (optional)"),
    from_time: str = Query(..., alias="from", description="Start time (ISO 8601)"),
    to_time: str = Query(..., alias="to", description="End time (ISO 8601)"),
    limit: int = Query(10, le=50, description="Max results"),
):
    """
    Search templates semantically.
    
    Uses embeddings to find templates matching the natural language query.
    """
    service = get_semantic_service()
    
    try:
        return await service.search(
            query=q,
            tenant_id=tenant_id,
            service_name=service_name or "",
            from_time=from_time,
            to_time=to_time,
            limit=limit,
        )
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
