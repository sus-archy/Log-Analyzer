"""Schemas package."""

from .ingest import (
    RawLogEvent,
    IngestRequest,
    IngestFolderRequest,
    IngestStats,
    IngestResponse,
)
from .logs import (
    NormalizedLogEvent,
    LogEventResponse,
    LogQueryParams,
    LogQueryResponse,
    map_severity,
    severity_name,
    SEVERITY_MAP,
    SEVERITY_NAMES,
)
from .templates import (
    LogTemplate,
    TemplateWithCount,
    TopTemplatesParams,
    TopTemplatesResponse,
    TemplateDetailParams,
    TemplateOccurrence,
    TemplateDetailResponse,
)
from .chat import (
    ChatRequest,
    ChatResponse,
    Citation,
    SemanticSearchParams,
    SemanticSearchResult,
    SemanticSearchResponse,
)

__all__ = [
    # Ingest
    "RawLogEvent",
    "IngestRequest",
    "IngestFolderRequest",
    "IngestStats",
    "IngestResponse",
    # Logs
    "NormalizedLogEvent",
    "LogEventResponse",
    "LogQueryParams",
    "LogQueryResponse",
    "map_severity",
    "severity_name",
    "SEVERITY_MAP",
    "SEVERITY_NAMES",
    # Templates
    "LogTemplate",
    "TemplateWithCount",
    "TopTemplatesParams",
    "TopTemplatesResponse",
    "TemplateDetailParams",
    "TemplateOccurrence",
    "TemplateDetailResponse",
    # Chat
    "ChatRequest",
    "ChatResponse",
    "Citation",
    "SemanticSearchParams",
    "SemanticSearchResult",
    "SemanticSearchResponse",
]
