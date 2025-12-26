"""
JSONL parser - parses JSON Lines format log files.
"""

import json
from typing import Any, Dict, Generator, Optional

from ..core.logging import get_logger
from ..core.time import parse_to_utc_iso, now_utc_iso
from ..schemas.logs import map_severity

logger = get_logger(__name__)

# Common field name mappings
TIMESTAMP_FIELDS = ["timestamp", "time", "datetime", "@timestamp", "ts", "date", "created_at"]
SEVERITY_FIELDS = ["severity", "level", "log_level", "loglevel", "priority"]
MESSAGE_FIELDS = ["message", "msg", "body", "log", "text", "content"]
SERVICE_FIELDS = ["service", "service_name", "serviceName", "app", "application"]
HOST_FIELDS = ["host", "hostname", "server", "machine", "node"]
TRACE_FIELDS = ["trace_id", "traceId", "trace", "request_id", "requestId"]
SPAN_FIELDS = ["span_id", "spanId", "span"]


def find_field(data: Dict[str, Any], field_names: list) -> Optional[Any]:
    """Find first matching field from a list of possible names."""
    for name in field_names:
        if name in data:
            return data[name]
        # Also check case-insensitive
        lower_name = name.lower()
        for key in data:
            if key.lower() == lower_name:
                return data[key]
    return None


def parse_jsonl_line(line: str, default_service: str = "unknown") -> Optional[Dict[str, Any]]:
    """
    Parse a single JSONL line into normalized fields.
    
    Args:
        line: JSON string
        default_service: Default service name if not in data
        
    Returns:
        Dict with normalized fields or None if parse fails
    """
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse JSON: {e}")
        return None
    
    if not isinstance(data, dict):
        return None
    
    # Extract timestamp
    raw_ts = find_field(data, TIMESTAMP_FIELDS)
    if raw_ts:
        timestamp_utc = parse_to_utc_iso(str(raw_ts))
    else:
        timestamp_utc = now_utc_iso()
    
    # Extract severity
    raw_severity = find_field(data, SEVERITY_FIELDS)
    severity = map_severity(raw_severity)
    
    # Extract message
    message = find_field(data, MESSAGE_FIELDS)
    if message is None:
        # Use entire JSON as message if no message field
        message = line
    message = str(message)
    
    # Extract service name
    service_name = find_field(data, SERVICE_FIELDS)
    if service_name:
        service_name = str(service_name)
    else:
        service_name = default_service
    
    # Extract host
    host = find_field(data, HOST_FIELDS)
    host = str(host) if host else ""
    
    # Extract trace/span
    trace_id = find_field(data, TRACE_FIELDS)
    trace_id = str(trace_id) if trace_id else ""
    
    span_id = find_field(data, SPAN_FIELDS)
    span_id = str(span_id) if span_id else ""
    
    # Collect remaining fields as attributes
    known_fields = set()
    for field_list in [TIMESTAMP_FIELDS, SEVERITY_FIELDS, MESSAGE_FIELDS, 
                       SERVICE_FIELDS, HOST_FIELDS, TRACE_FIELDS, SPAN_FIELDS]:
        known_fields.update(f.lower() for f in field_list)
    
    attributes = {
        k: v for k, v in data.items() 
        if k.lower() not in known_fields
    }
    
    return {
        "timestamp_utc": timestamp_utc,
        "severity": severity,
        "message": message,
        "service_name": service_name,
        "host": host,
        "trace_id": trace_id,
        "span_id": span_id,
        "attributes": attributes,
        "body_raw": line,
    }


def parse_jsonl_file(
    lines: Generator,
    default_service: str = "unknown",
) -> Generator[Dict[str, Any], None, None]:
    """
    Parse JSONL file lines.
    
    Args:
        lines: Generator of (line_num, line) tuples
        default_service: Default service name
        
    Yields:
        Normalized log event dicts
    """
    for line_num, line in lines:
        result = parse_jsonl_line(line, default_service)
        if result:
            yield result
