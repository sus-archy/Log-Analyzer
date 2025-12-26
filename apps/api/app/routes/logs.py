"""
Logs routes for querying log events.
"""

from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..core.config import settings
from ..core.logging import get_logger
from ..core.cache import get_cache, CacheTTL
from ..schemas.logs import LogQueryResponse, LogEventResponse
from ..services.query_service import get_query_service

logger = get_logger(__name__)
router = APIRouter(prefix="/logs", tags=["logs"])


class FileInfo(BaseModel):
    name: str
    path: str
    size: int
    is_dir: bool


class BrowseResponse(BaseModel):
    current_path: str
    files: List[FileInfo]


@router.get("", response_model=LogQueryResponse)
async def query_logs(
    tenant_id: Optional[str] = Query(None, description="Tenant ID"),
    service_name: Optional[str] = Query(None, description="Service name filter"),
    from_time: str = Query(..., alias="from", description="Start time (ISO 8601)"),
    to_time: str = Query(..., alias="to", description="End time (ISO 8601)"),
    severity_min: Optional[int] = Query(None, ge=0, le=5, description="Minimum severity"),
    template_hash: Optional[int] = Query(None, description="Filter by template"),
    trace_id: Optional[str] = Query(None, description="Filter by trace ID"),
    limit: int = Query(200, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """
    Query logs with filters.
    
    Time range (from/to) is required.
    """
    service = get_query_service()
    
    try:
        return await service.query_logs(
            tenant_id=tenant_id,
            from_time=from_time,
            to_time=to_time,
            service_name=service_name,
            severity_min=severity_min,
            template_hash=template_hash,
            trace_id=trace_id,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{log_id}", response_model=LogEventResponse)
async def get_log(
    log_id: int,
    tenant_id: Optional[str] = Query(None, description="Tenant ID"),
):
    """Get a single log by ID."""
    service = get_query_service()
    
    try:
        log = await service.get_log_by_id(tenant_id=tenant_id, log_id=log_id)
        if log is None:
            raise HTTPException(status_code=404, detail="Log not found")
        return log
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get log failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services/list")
async def list_services(
    tenant_id: Optional[str] = Query(None, description="Tenant ID"),
):
    """Get list of all service names."""
    cache = get_cache()
    cache_key = f"services_list:{tenant_id or 'all'}"
    
    # Try cache first
    cached = await cache.get(cache_key)
    if cached is not None:
        return cached
    
    service = get_query_service()
    
    try:
        services = await service.get_services(tenant_id=tenant_id)
        result = {"services": services}
        await cache.set(cache_key, result, CacheTTL.SERVICES_LIST)
        return result
    except Exception as e:
        logger.error(f"List services failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/quick")
async def get_quick_stats(
    tenant_id: Optional[str] = Query(None, description="Tenant ID"),
):
    """
    Get quick database statistics without expensive COUNT queries.
    Uses SQLite's internal stats for fast response.
    """
    cache = get_cache()
    cache_key = f"quick_stats:{tenant_id or 'all'}"
    
    # Try cache first
    cached = await cache.get(cache_key)
    if cached is not None:
        return cached
    
    service = get_query_service()
    
    try:
        stats = await service.get_quick_stats(tenant_id=tenant_id)
        await cache.set(cache_key, stats, CacheTTL.QUICK_STATS)
        return stats
    except Exception as e:
        logger.error(f"Quick stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/browse", response_model=BrowseResponse)
async def browse_log_files(
    path: Optional[str] = Query(None, description="Relative path within logs folder"),
):
    """
    Browse log files in the logs folder.
    
    Returns list of files and directories.
    """
    base_path = settings.logs_folder_resolved
    
    if path:
        # Sanitize path to prevent directory traversal
        clean_path = Path(path).as_posix().lstrip("/")
        target_path = base_path / clean_path
        # Ensure we're still within the logs folder
        try:
            target_path.resolve().relative_to(base_path.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid path")
    else:
        target_path = base_path
    
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    
    if not target_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    files: List[FileInfo] = []
    valid_extensions = {'.log', '.txt', '.jsonl', '.csv'}
    
    try:
        for item in sorted(target_path.iterdir()):
            if item.name.startswith('.'):
                continue
            
            is_dir = item.is_dir()
            if not is_dir and item.suffix.lower() not in valid_extensions:
                continue
            
            rel_path = item.relative_to(base_path).as_posix()
            
            files.append(FileInfo(
                name=item.name,
                path=rel_path,
                size=item.stat().st_size if not is_dir else 0,
                is_dir=is_dir,
            ))
        
        # Sort: directories first, then files
        files.sort(key=lambda x: (not x.is_dir, x.name.lower()))
        
        current_rel = target_path.relative_to(base_path).as_posix() if target_path != base_path else ""
        
        return BrowseResponse(
            current_path=current_rel,
            files=files,
        )
    except Exception as e:
        logger.error(f"Browse failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/files/ingest")
async def ingest_selected_files(
    files: List[str],
    tenant_id: Optional[str] = None,
):
    """
    Ingest selected files from the logs folder.
    
    Files should be relative paths within the logs folder.
    """
    from ..services.ingest_service import get_ingest_service
    from ..schemas.ingest import IngestStats
    
    base_path = settings.logs_folder_resolved
    service = get_ingest_service()
    
    total_stats = IngestStats(
        files_processed=0,
        lines_processed=0,
        events_inserted=0,
        templates_discovered=0,
        errors=[],
    )
    
    for file_path in files:
        # Sanitize path
        clean_path = Path(file_path).as_posix().lstrip("/")
        full_path = base_path / clean_path
        
        # Security check
        try:
            full_path.resolve().relative_to(base_path.resolve())
        except ValueError:
            total_stats.errors.append(f"Invalid path: {file_path}")
            continue
        
        if not full_path.exists() or full_path.is_dir():
            total_stats.errors.append(f"File not found: {file_path}")
            continue
        
        try:
            stats = await service.ingest_single_file(
                file_path=full_path,
                tenant_id=tenant_id,
            )
            total_stats.files_processed += stats.files_processed
            total_stats.lines_processed += stats.lines_processed
            total_stats.events_inserted += stats.events_inserted
            total_stats.templates_discovered += stats.templates_discovered
        except Exception as e:
            total_stats.errors.append(f"Error processing {file_path}: {e}")
    
    return {
        "success": total_stats.files_processed > 0,
        "stats": total_stats,
        "message": f"Processed {total_stats.files_processed} files, {total_stats.events_inserted} events",
    }
