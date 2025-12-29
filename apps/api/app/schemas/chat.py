"""
Chat schemas for grounded chat with logs.
"""

import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""
    
    tenant_id: Optional[str] = None
    service_name: str = Field(..., description="Service to analyze", max_length=100)
    from_time: str = Field(..., alias="from", description="Start time (ISO 8601)")
    to_time: str = Field(..., alias="to", description="End time (ISO 8601)")
    question: str = Field(
        ..., 
        min_length=1, 
        max_length=5000,  # Limit question length
        description="User question"
    )
    
    class Config:
        populate_by_name = True
    
    @field_validator('question')
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        """Basic sanitization of user question."""
        # Remove any null bytes
        v = v.replace('\x00', '')
        # Limit consecutive whitespace
        v = re.sub(r'\s+', ' ', v).strip()
        return v
    
    @field_validator('service_name')
    @classmethod
    def sanitize_service_name(cls, v: str) -> str:
        """Validate service name format."""
        # Only allow alphanumeric, dash, underscore, dot
        if v and not re.match(r'^[\w\-\.]+$', v):
            # Remove invalid characters
            v = re.sub(r'[^\w\-\.]', '', v)
        return v


class Citation(BaseModel):
    """Citation reference to evidence."""
    
    type: str = Field(..., description="'template' or 'log'")
    service_name: Optional[str] = None
    template_hash: Optional[int] = None
    template_text: Optional[str] = None
    log_id: Optional[int] = None
    relevance: Optional[str] = None  # why this is relevant


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    
    answer: str = Field(..., description="Grounded answer")
    citations: List[Citation] = Field(default_factory=list, description="Evidence citations")
    confidence: str = Field(default="medium", description="low/medium/high")
    next_steps: List[str] = Field(default_factory=list, description="Suggested next queries")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SemanticSearchParams(BaseModel):
    """Parameters for semantic search."""
    
    tenant_id: Optional[str] = None
    service_name: str
    from_time: str = Field(..., alias="from")
    to_time: str = Field(..., alias="to")
    q: str = Field(..., min_length=1, description="Search query")
    limit: int = Field(default=10, le=50)
    
    class Config:
        populate_by_name = True


class SemanticSearchResult(BaseModel):
    """Single semantic search result."""
    
    template_hash: str  # String to prevent JS precision loss
    template_text: str
    score: float = Field(..., description="Similarity score")
    count: int = Field(..., description="Occurrences in time window")
    sample_log_ids: List[int] = Field(default_factory=list)


class SemanticSearchResponse(BaseModel):
    """Response from semantic search."""
    
    results: List[SemanticSearchResult]
    query: str
    service_name: str
    from_time: str
    to_time: str
