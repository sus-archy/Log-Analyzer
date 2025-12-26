"""
Ingest routes for log ingestion.
"""

from typing import List, Optional, Union
from pathlib import Path
import tempfile
import shutil

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from ..core.config import settings
from ..core.logging import get_logger
from ..schemas.ingest import (
    RawLogEvent,
    IngestRequest,
    IngestFolderRequest,
    IngestResponse,
)
from ..services.ingest_service import get_ingest_service

logger = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
async def ingest_events(request: IngestRequest):
    """
    Ingest log events.
    
    Accepts a single event or list of events.
    """
    service = get_ingest_service()
    
    # Normalize to list
    events = request.events
    if isinstance(events, RawLogEvent):
        events = [events]
    
    # Convert to dicts
    events_dicts = [e.model_dump() for e in events]
    
    try:
        stats = await service.ingest_events(
            events=events_dicts,
            tenant_id=request.tenant_id,
        )
        
        return IngestResponse(
            success=True,
            stats=stats,
            message=f"Ingested {stats.events_inserted} events",
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/from_folder", response_model=IngestResponse)
async def ingest_from_folder(request: Optional[IngestFolderRequest] = None):
    """
    Ingest all log files from the logs folder.
    
    Reads all supported files (.log, .txt, .jsonl, .csv) recursively.
    """
    service = get_ingest_service()
    
    folder_path = None
    tenant_id = None
    
    if request:
        if request.folder_path:
            folder_path = Path(request.folder_path)
        tenant_id = request.tenant_id
    
    try:
        stats = await service.ingest_from_folder(
            folder_path=folder_path,
            tenant_id=tenant_id,
        )
        
        return IngestResponse(
            success=True,
            stats=stats,
            message=f"Processed {stats.files_processed} files, ingested {stats.events_inserted} events",
        )
    except Exception as e:
        logger.error(f"Folder ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=IngestResponse)
async def upload_log_file(
    file: UploadFile = File(...),
    service_name: Optional[str] = Form(None),
    tenant_id: Optional[str] = Form(None),
):
    """
    Upload and ingest a log file.
    
    Accepts .log, .txt, .jsonl, .csv files.
    Service name is derived from filename if not provided.
    """
    service = get_ingest_service()
    
    # Validate file extension
    valid_extensions = {'.log', '.txt', '.jsonl', '.csv'}
    file_ext = Path(file.filename or "file.log").suffix.lower()
    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported: {', '.join(valid_extensions)}"
        )
    
    # Save to temp file
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=file_ext,
            prefix=f"{service_name or 'uploaded'}_"
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
        
        # Ingest the file
        stats = await service.ingest_single_file(
            file_path=tmp_path,
            service_name=service_name,
            tenant_id=tenant_id,
        )
        
        return IngestResponse(
            success=True,
            stats=stats,
            message=f"Uploaded and ingested {file.filename}: {stats.events_inserted} events",
        )
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()
