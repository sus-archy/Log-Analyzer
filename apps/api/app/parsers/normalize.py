"""
Log normalizer - converts parsed log data to normalized events.
"""

import json
from typing import Any, Dict, List, Optional

from ..core.config import settings
from ..core.logging import get_logger
from ..core.time import now_utc_iso
from ..schemas.logs import NormalizedLogEvent, map_severity

logger = get_logger(__name__)


def normalize_event(
    parsed: Dict[str, Any],
    tenant_id: Optional[str] = None,
    template_hash: int = 0,
    parameters: Optional[List[str]] = None,
    environment: str = "prod",
) -> NormalizedLogEvent:
    """
    Normalize a parsed log event to the canonical format.
    
    Args:
        parsed: Parsed log data dict
        tenant_id: Override tenant ID
        template_hash: Computed template hash
        parameters: Extracted parameters from template mining
        environment: Environment name
        
    Returns:
        NormalizedLogEvent ready for storage
    """
    now = now_utc_iso()
    
    # Get tenant ID
    if tenant_id is None:
        tenant_id = settings.tenant_id_default
    
    # Ensure service name
    service_name = parsed.get("service_name", "unknown")
    if not service_name:
        service_name = "unknown"
    
    # Ensure timestamp
    timestamp_utc = parsed.get("timestamp_utc", now)
    if not timestamp_utc:
        timestamp_utc = now
    
    # Ensure severity
    severity = parsed.get("severity", 2)
    if isinstance(severity, str):
        severity = map_severity(severity)
    severity = max(0, min(5, int(severity)))
    
    # Serialize parameters and attributes
    if parameters is None:
        parameters = []
    
    attributes = parsed.get("attributes", {})
    if not isinstance(attributes, dict):
        attributes = {}
    
    return NormalizedLogEvent(
        tenant_id=tenant_id,
        service_name=service_name,
        environment=environment,
        timestamp_utc=timestamp_utc,
        ingest_timestamp_utc=now,
        severity=severity,
        host=parsed.get("host", "") or "",
        template_hash=template_hash,
        parameters_json=json.dumps(parameters),
        trace_id=parsed.get("trace_id", "") or "",
        span_id=parsed.get("span_id", "") or "",
        attributes_json=json.dumps(attributes),
        body_raw=parsed.get("body_raw", "") or parsed.get("message", ""),
    )


def normalize_raw_event(
    message: str,
    tenant_id: Optional[str] = None,
    service_name: str = "unknown",
    timestamp: Optional[str] = None,
    severity: Optional[int] = None,
    host: str = "",
    trace_id: str = "",
    span_id: str = "",
    attributes: Optional[Dict[str, Any]] = None,
    environment: str = "prod",
    template_hash: int = 0,
    parameters: Optional[List[str]] = None,
) -> NormalizedLogEvent:
    """
    Create a normalized event from raw inputs.
    
    Convenience function for API ingestion.
    """
    now = now_utc_iso()
    
    if tenant_id is None:
        tenant_id = settings.tenant_id_default
    
    if timestamp is None:
        timestamp = now
    
    if severity is None:
        severity = 2  # INFO
    
    if attributes is None:
        attributes = {}
    
    if parameters is None:
        parameters = []
    
    return NormalizedLogEvent(
        tenant_id=tenant_id,
        service_name=service_name,
        environment=environment,
        timestamp_utc=timestamp,
        ingest_timestamp_utc=now,
        severity=severity,
        host=host,
        template_hash=template_hash,
        parameters_json=json.dumps(parameters),
        trace_id=trace_id,
        span_id=span_id,
        attributes_json=json.dumps(attributes),
        body_raw=message,
    )
