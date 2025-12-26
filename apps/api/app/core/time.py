"""
Time parsing and formatting utilities.
Handles timestamp parsing and conversion to UTC ISO 8601 format.
"""

import re
from datetime import datetime, timezone
from typing import Optional, Tuple

from dateutil import parser as dateutil_parser


# Common timestamp patterns
TIMESTAMP_PATTERNS = [
    # ISO 8601
    (r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?", None),
    # Apache/NCSA common log format: [10/Oct/2000:13:55:36 -0700]
    (r"\[\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4}\]", "[%d/%b/%Y:%H:%M:%S %z]"),
    # Syslog: Dec 22 10:15:30
    (r"[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}", "%b %d %H:%M:%S"),
    # Unix timestamp (epoch)
    (r"^\d{10}(?:\.\d+)?$", "epoch"),
    # Unix timestamp in milliseconds
    (r"^\d{13}$", "epoch_ms"),
    # Common log format: 10/Oct/2000:13:55:36
    (r"\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}", "%d/%b/%Y:%H:%M:%S"),
    # YYYY-MM-DD HH:MM:SS
    (r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "%Y-%m-%d %H:%M:%S"),
    # MM/DD/YYYY HH:MM:SS
    (r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}", "%m/%d/%Y %H:%M:%S"),
]


def extract_timestamp(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract timestamp from text using pattern matching.
    
    Args:
        text: Log message text
        
    Returns:
        Tuple of (extracted timestamp string, remaining text) or (None, original text)
    """
    for pattern, _ in TIMESTAMP_PATTERNS:
        match = re.search(pattern, text)
        if match:
            ts_str = match.group()
            remaining = text[:match.start()] + text[match.end():]
            return ts_str, remaining.strip()
    return None, text


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """
    Parse a timestamp string into a datetime object.
    
    Args:
        ts_str: Timestamp string in various formats
        
    Returns:
        datetime object or None if parsing fails
    """
    if not ts_str:
        return None
    
    ts_str = ts_str.strip()
    
    # Handle epoch timestamps
    if re.match(r"^\d{10}(?:\.\d+)?$", ts_str):
        try:
            return datetime.fromtimestamp(float(ts_str), tz=timezone.utc)
        except (ValueError, OSError):
            pass
    
    # Handle epoch milliseconds
    if re.match(r"^\d{13}$", ts_str):
        try:
            return datetime.fromtimestamp(int(ts_str) / 1000, tz=timezone.utc)
        except (ValueError, OSError):
            pass
    
    # Try pattern-based parsing
    for pattern, fmt in TIMESTAMP_PATTERNS:
        if fmt and fmt not in ("epoch", "epoch_ms"):
            match = re.search(pattern, ts_str)
            if match:
                try:
                    ts = ts_str if match.group() == ts_str else match.group()
                    # Remove brackets if present
                    ts = ts.strip("[]")
                    dt = datetime.strptime(ts, fmt.strip("[]"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except ValueError:
                    continue
    
    # Fall back to dateutil parser
    try:
        dt = dateutil_parser.parse(ts_str, fuzzy=True)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, dateutil_parser.ParserError):
        return None


def to_utc_iso(dt: Optional[datetime]) -> str:
    """
    Convert datetime to UTC ISO 8601 string.
    
    Args:
        dt: datetime object
        
    Returns:
        ISO 8601 formatted string in UTC
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    
    # Convert to UTC if timezone-aware
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    else:
        dt = dt.replace(tzinfo=timezone.utc)
    
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def now_utc_iso() -> str:
    """Get current UTC time as ISO 8601 string."""
    return to_utc_iso(datetime.now(timezone.utc))


def parse_to_utc_iso(ts_str: str) -> str:
    """
    Parse timestamp string and return UTC ISO 8601 format.
    Falls back to current time if parsing fails.
    
    Args:
        ts_str: Timestamp string
        
    Returns:
        UTC ISO 8601 formatted string
    """
    dt = parse_timestamp(ts_str)
    return to_utc_iso(dt)
