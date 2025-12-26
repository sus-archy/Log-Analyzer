"""
Ingest schemas for log event ingestion.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class RawLogEvent(BaseModel):
    """Raw log event as received from external sources."""
    
    message: str = Field(..., description="Raw log message")
    timestamp: Optional[str] = Field(None, description="Timestamp string")
    severity: Optional[Union[str, int]] = Field(None, description="Severity level")
    service_name: Optional[str] = Field(None, description="Service name")
    host: Optional[str] = Field(None, description="Host name")
    environment: Optional[str] = Field(None, description="Environment (prod/staging/dev)")
    trace_id: Optional[str] = Field(None, description="Trace ID")
    span_id: Optional[str] = Field(None, description="Span ID")
    attributes: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional attributes")


class IngestRequest(BaseModel):
    """Request body for /ingest endpoint."""
    
    events: Union[RawLogEvent, List[RawLogEvent]] = Field(
        ...,
        description="Single event or list of events to ingest"
    )
    tenant_id: Optional[str] = Field(None, description="Tenant ID override")


class IngestFolderRequest(BaseModel):
    """Request body for /ingest_from_folder endpoint."""
    
    folder_path: Optional[str] = Field(
        None,
        description="Folder path override (defaults to ./logs/)"
    )
    tenant_id: Optional[str] = Field(None, description="Tenant ID override")


class IngestStats(BaseModel):
    """Statistics from ingestion operation."""
    
    files_processed: int = Field(..., description="Number of files processed")
    files_skipped: int = Field(default=0, description="Number of duplicate files skipped")
    lines_processed: int = Field(..., description="Number of lines processed")
    events_inserted: int = Field(..., description="Number of events inserted")
    templates_discovered: int = Field(..., description="Number of new templates")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    skipped_files: List[str] = Field(default_factory=list, description="Names of skipped duplicate files")


class IngestResponse(BaseModel):
    """Response from ingestion endpoints."""
    
    success: bool
    stats: IngestStats
    message: str = Field(default="Ingestion completed")
