"""
Log event schemas.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NormalizedLogEvent(BaseModel):
    """Normalized log event with all required fields."""
    
    tenant_id: str
    service_name: str
    environment: str = "prod"
    
    timestamp_utc: str = Field(..., description="ISO 8601 UTC timestamp")
    ingest_timestamp_utc: str = Field(..., description="Ingestion time")
    
    severity: int = Field(..., ge=0, le=5, description="Severity 0-5")
    
    host: str = ""
    
    template_hash: int = Field(..., description="Template hash (64-bit)")
    parameters_json: str = Field(default="[]", description="JSON array of parameters")
    
    trace_id: str = ""
    span_id: str = ""
    
    attributes_json: str = Field(default="{}", description="JSON attributes map")
    body_raw: str = Field(default="", description="Original raw message")


class LogEventResponse(BaseModel):
    """Log event as returned by API."""
    
    id: int
    tenant_id: str
    service_name: str
    environment: str
    
    timestamp_utc: str
    severity: int
    severity_name: str
    
    host: str
    
    template_hash: int
    template_text: Optional[str] = None
    parameters: List[Any]
    
    trace_id: str
    span_id: str
    
    attributes: Dict[str, Any]
    body_raw: str


class LogQueryParams(BaseModel):
    """Query parameters for log search."""
    
    tenant_id: str
    service_name: Optional[str] = None
    from_time: str = Field(..., alias="from", description="Start time (ISO 8601)")
    to_time: str = Field(..., alias="to", description="End time (ISO 8601)")
    severity_min: Optional[int] = Field(None, ge=0, le=5)
    template_hash: Optional[int] = None
    trace_id: Optional[str] = None
    limit: int = Field(default=200, le=1000)
    offset: int = Field(default=0, ge=0)
    
    class Config:
        populate_by_name = True


class LogQueryResponse(BaseModel):
    """Response from log query."""
    
    logs: List[LogEventResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


# Severity mapping
SEVERITY_MAP = {
    "trace": 0,
    "debug": 1,
    "info": 2,
    "warn": 3,
    "warning": 3,
    "error": 4,
    "err": 4,
    "fatal": 5,
    "critical": 5,
    "crit": 5,
    "emerg": 5,
    "emergency": 5,
}

SEVERITY_NAMES = {
    0: "TRACE",
    1: "DEBUG", 
    2: "INFO",
    3: "WARN",
    4: "ERROR",
    5: "FATAL",
}


def map_severity(value: Optional[str | int]) -> int:
    """
    Map severity value to integer 0-5.
    
    Args:
        value: Severity as string or int
        
    Returns:
        Integer severity (0-5), defaults to 2 (INFO)
    """
    if value is None:
        return 2  # INFO
    
    if isinstance(value, int):
        return max(0, min(5, value))
    
    if isinstance(value, str):
        # Try to parse as int
        try:
            return max(0, min(5, int(value)))
        except ValueError:
            pass
        
        # Map string to int
        return SEVERITY_MAP.get(value.lower().strip(), 2)
    
    return 2  # INFO default


def severity_name(level: int) -> str:
    """Get severity name from level."""
    return SEVERITY_NAMES.get(level, "UNKNOWN")
