"""
Templates routes for querying log templates.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..core.logging import get_logger
from ..schemas.templates import TopTemplatesResponse, TemplateDetailResponse
from ..services.query_service import get_query_service

logger = get_logger(__name__)
router = APIRouter(prefix="/templates", tags=["templates"])


@router.get("/top", response_model=TopTemplatesResponse)
async def get_top_templates(
    tenant_id: Optional[str] = Query(None, description="Tenant ID"),
    service_name: str = Query("", description="Service name (empty for all services)"),
    from_time: str = Query(..., alias="from", description="Start time (ISO 8601)"),
    to_time: str = Query(..., alias="to", description="End time (ISO 8601)"),
    severity_min: int = Query(0, ge=0, le=5, description="Minimum severity (default 0=all)"),
    limit: int = Query(50, le=100, description="Max templates"),
):
    """
    Get top templates by occurrence count.
    
    Service name and time range are required.
    """
    service = get_query_service()
    
    try:
        return await service.get_top_templates(
            tenant_id=tenant_id,
            service_name=service_name,
            from_time=from_time,
            to_time=to_time,
            severity_min=severity_min,
            limit=limit,
        )
    except Exception as e:
        logger.error(f"Get top templates failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{template_hash}", response_model=TemplateDetailResponse)
async def get_template_detail(
    template_hash: int,
    tenant_id: Optional[str] = Query(None, description="Tenant ID"),
    service_name: str = Query(..., description="Service name (required)"),
    from_time: str = Query(..., alias="from", description="Start time (ISO 8601)"),
    to_time: str = Query(..., alias="to", description="End time (ISO 8601)"),
):
    """
    Get detailed template information with occurrences.
    
    Service name and time range are required.
    """
    service = get_query_service()
    
    try:
        detail = await service.get_template_detail(
            tenant_id=tenant_id,
            service_name=service_name,
            template_hash=template_hash,
            from_time=from_time,
            to_time=to_time,
        )
        
        if detail is None:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return detail
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get template detail failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
