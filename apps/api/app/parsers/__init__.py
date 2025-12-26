"""Parsers package."""

from .folder_loader import (
    discover_log_files,
    infer_service_name_from_path,
    read_file_lines,
    get_file_format,
    SUPPORTED_EXTENSIONS,
)
from .jsonl_parser import parse_jsonl_line, parse_jsonl_file
from .text_parser import (
    parse_text_line,
    parse_text_file,
    parse_csv_structured_line,
    extract_severity,
)
from .normalize import normalize_event, normalize_raw_event
from .drain_miner import DrainMiner, compute_template_hash

__all__ = [
    # Folder loader
    "discover_log_files",
    "infer_service_name_from_path",
    "read_file_lines",
    "get_file_format",
    "SUPPORTED_EXTENSIONS",
    # JSONL parser
    "parse_jsonl_line",
    "parse_jsonl_file",
    # Text parser
    "parse_text_line",
    "parse_text_file",
    "parse_csv_structured_line",
    "extract_severity",
    # Normalizer
    "normalize_event",
    "normalize_raw_event",
    # Drain miner
    "DrainMiner",
    "compute_template_hash",
]
