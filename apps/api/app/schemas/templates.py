"""
Template schemas.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class LogTemplate(BaseModel):
    """Log template record."""
    
    tenant_id: str
    service_name: str
    template_hash: str  # String to prevent JS precision loss
    template_text: str
    
    first_seen_utc: str
    last_seen_utc: str
    
    embedding_state: str = "none"  # none|queued|ready|failed
    embedding_model: str = ""
    embedding_updated_utc: str = ""


class TemplateWithCount(BaseModel):
    """Template with occurrence count."""
    
    template_hash: str  # String to prevent JS precision loss
    template_text: str
    count: int
    sample_log_id: Optional[int] = None
    first_seen_utc: str
    last_seen_utc: str


class TopTemplatesParams(BaseModel):
    """Parameters for top templates query."""
    
    tenant_id: str
    service_name: str
    from_time: str = Field(..., alias="from")
    to_time: str = Field(..., alias="to")
    severity_min: int = Field(default=4, ge=0, le=5)  # Default to ERROR
    limit: int = Field(default=50, le=100)
    
    class Config:
        populate_by_name = True


class TopTemplatesResponse(BaseModel):
    """Response for top templates."""
    
    templates: List[TemplateWithCount]
    service_name: str
    from_time: str
    to_time: str
    total: int


class TemplateDetailParams(BaseModel):
    """Parameters for template detail query."""
    
    tenant_id: str
    service_name: str
    from_time: str = Field(..., alias="from")
    to_time: str = Field(..., alias="to")
    
    class Config:
        populate_by_name = True


class TemplateOccurrence(BaseModel):
    """A single occurrence of a template."""
    
    log_id: int
    timestamp_utc: str
    parameters: List[str]
    severity: int
    host: str


class TemplateDetailResponse(BaseModel):
    """Detailed template information."""
    
    template_hash: str  # String to prevent JS precision loss
    template_text: str
    service_name: str
    
    first_seen_utc: str
    last_seen_utc: str
    
    total_count: int
    occurrences: List[TemplateOccurrence]
    
    embedding_state: str
