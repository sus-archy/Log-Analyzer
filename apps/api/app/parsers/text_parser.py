"""
Text parser - parses plain text log files (.log, .txt).
"""

import re
from typing import Any, Dict, Generator, Optional

from ..core.logging import get_logger
from ..core.time import extract_timestamp, parse_to_utc_iso, now_utc_iso
from ..schemas.logs import map_severity, SEVERITY_MAP

logger = get_logger(__name__)

# Severity patterns to search for in log lines
SEVERITY_PATTERNS = [
    # Standard severity words
    r"\b(TRACE|DEBUG|INFO|WARN(?:ING)?|ERROR|ERR|FATAL|CRIT(?:ICAL)?|EMERG(?:ENCY)?)\b",
    # Bracketed severity
    r"\[(TRACE|DEBUG|INFO|WARN(?:ING)?|ERROR|ERR|FATAL|CRIT(?:ICAL)?|EMERG(?:ENCY)?)\]",
    # Log4j style
    r"^\s*\d+\s+(TRACE|DEBUG|INFO|WARN|ERROR|FATAL)\s+",
]

# Compiled regex for efficiency
_severity_regexes = [re.compile(p, re.IGNORECASE) for p in SEVERITY_PATTERNS]


def extract_severity(text: str) -> tuple[int, str]:
    """
    Extract severity level from log text.
    
    Args:
        text: Log line text
        
    Returns:
        Tuple of (severity_int, remaining_text)
    """
    for regex in _severity_regexes:
        match = regex.search(text)
        if match:
            severity_str = match.group(1).upper()
            severity = map_severity(severity_str)
            # Don't remove severity from text - keep original
            return severity, text
    
    # Default to INFO
    return 2, text


def parse_text_line(
    line: str,
    default_service: str = "unknown",
) -> Dict[str, Any]:
    """
    Parse a single text log line into normalized fields.
    
    Args:
        line: Log line text
        default_service: Default service name
        
    Returns:
        Dict with normalized fields
    """
    # Extract timestamp
    ts_str, message = extract_timestamp(line)
    if ts_str:
        timestamp_utc = parse_to_utc_iso(ts_str)
    else:
        timestamp_utc = now_utc_iso()
        message = line
    
    # Ensure message is not None
    if message is None:
        message = line
    
    # Extract severity
    severity, _ = extract_severity(message)
    
    # Clean up message
    message = message.strip()
    if not message:
        message = line
    
    return {
        "timestamp_utc": timestamp_utc,
        "severity": severity,
        "message": message,
        "service_name": default_service,
        "host": "",
        "trace_id": "",
        "span_id": "",
        "attributes": {},
        "body_raw": line,
    }


def parse_text_file(
    lines: Generator,
    default_service: str = "unknown",
) -> Generator[Dict[str, Any], None, None]:
    """
    Parse text log file lines.
    
    Args:
        lines: Generator of (line_num, line) tuples
        default_service: Default service name
        
    Yields:
        Normalized log event dicts
    """
    for line_num, line in lines:
        result = parse_text_line(line, default_service)
        yield result


def parse_csv_structured_line(
    row: Dict[str, str],
    default_service: str = "unknown",
) -> Dict[str, Any]:
    """
    Parse a CSV row from structured log files (like loghub).
    
    These files have columns like: LineId, Date, Time, Content, EventTemplate, etc.
    
    Args:
        row: Dict from CSV DictReader
        default_service: Default service name
        
    Returns:
        Dict with normalized fields
    """
    # Try to get content/message
    message = (
        row.get("Content") or 
        row.get("content") or 
        row.get("Message") or 
        row.get("message") or
        row.get("RawLog") or
        ""
    )
    
    # Try to get timestamp
    date = row.get("Date") or row.get("date") or ""
    time = row.get("Time") or row.get("time") or ""
    
    if date and time:
        ts_str = f"{date} {time}"
        timestamp_utc = parse_to_utc_iso(ts_str)
    elif date:
        timestamp_utc = parse_to_utc_iso(date)
    else:
        timestamp_utc = now_utc_iso()
    
    # Try to get severity
    level = (
        row.get("Level") or 
        row.get("level") or 
        row.get("Severity") or
        row.get("severity") or
        ""
    )
    severity = map_severity(level) if level else 2
    
    # Try to extract severity from content if not found
    if not level and message:
        severity, _ = extract_severity(message)
    
    # Get component/service if available
    component = (
        row.get("Component") or
        row.get("component") or
        row.get("Service") or
        row.get("service") or
        ""
    )
    service_name = component if component else default_service
    
    # Get node/host
    host = (
        row.get("Node") or
        row.get("node") or
        row.get("Host") or
        row.get("host") or
        ""
    )
    
    # Collect other fields as attributes
    skip_fields = {
        "lineid", "date", "time", "content", "message", "level", "severity",
        "component", "service", "node", "host", "eventtemplate", "eventid",
        "parameterlist", "rawlog"
    }
    
    attributes = {
        k: v for k, v in row.items()
        if k.lower() not in skip_fields and v
    }
    
    return {
        "timestamp_utc": timestamp_utc,
        "severity": severity,
        "message": message,
        "service_name": service_name,
        "host": host,
        "trace_id": "",
        "span_id": "",
        "attributes": attributes,
        "body_raw": message,
    }
