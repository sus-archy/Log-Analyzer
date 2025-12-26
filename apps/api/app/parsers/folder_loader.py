"""
Folder loader - recursively loads log files from a directory.
"""

import os
from pathlib import Path
from typing import Generator, List, Tuple

from ..core.logging import get_logger

logger = get_logger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".log", ".txt", ".jsonl", ".csv"}

# Max file size in bytes (500MB - allow large files, skip only massive ones)
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024


def discover_log_files(folder_path: Path, max_file_size: int = MAX_FILE_SIZE_BYTES) -> List[Path]:
    """
    Recursively discover all log files in a folder.
    
    Args:
        folder_path: Root folder to search
        max_file_size: Skip files larger than this (bytes), 0 for no limit
        
    Returns:
        List of paths to log files
    """
    if not folder_path.exists():
        logger.warning(f"Folder does not exist: {folder_path}")
        return []
    
    files = []
    skipped = []
    
    for root, dirs, filenames in os.walk(folder_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        
        for filename in filenames:
            # Skip hidden files
            if filename.startswith("."):
                continue
            
            filepath = Path(root) / filename
            ext = filepath.suffix.lower()
            
            if ext in SUPPORTED_EXTENSIONS:
                # Check file size
                if max_file_size > 0:
                    file_size = filepath.stat().st_size
                    if file_size > max_file_size:
                        skipped.append((filepath, file_size))
                        continue
                files.append(filepath)
    
    if skipped:
        logger.info(f"Skipped {len(skipped)} large files (>{max_file_size/1024/1024:.1f}MB):")
        for path, size in skipped[:5]:
            logger.info(f"  - {path.name}: {size/1024/1024:.1f}MB")
        if len(skipped) > 5:
            logger.info(f"  ... and {len(skipped) - 5} more")
    
    logger.info(f"Discovered {len(files)} log files in {folder_path}")
    return sorted(files)


def infer_service_name_from_path(filepath: Path, base_folder: Path) -> str:
    """
    Infer service name from file path.
    
    Rules:
    1. Use first directory under base_folder as service name
    2. If file is directly in base_folder, use filename without extension
    
    Args:
        filepath: Path to the log file
        base_folder: Base logs folder
        
    Returns:
        Inferred service name
    """
    try:
        relative_path = filepath.relative_to(base_folder)
        parts = relative_path.parts
        
        if len(parts) > 1:
            # Use first directory as service name
            return parts[0].lower().replace(" ", "_").replace("-", "_")
        else:
            # Use filename without extension
            return filepath.stem.lower().replace(" ", "_").replace("-", "_")
    except ValueError:
        # File not under base_folder
        return "unknown"


def read_file_lines(filepath: Path) -> Generator[Tuple[int, str], None, None]:
    """
    Read a file line by line.
    
    Args:
        filepath: Path to file
        
    Yields:
        Tuples of (line_number, line_content)
    """
    encodings = ["utf-8", "latin-1", "cp1252"]
    
    for encoding in encodings:
        try:
            with open(filepath, "r", encoding=encoding, errors="replace") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.rstrip("\n\r")
                    if line.strip():  # Skip empty lines
                        yield (line_num, line)
            return
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return
    
    logger.warning(f"Could not read file with any encoding: {filepath}")


def get_file_format(filepath: Path) -> str:
    """
    Determine file format from extension.
    
    Returns:
        'jsonl', 'csv', or 'text'
    """
    ext = filepath.suffix.lower()
    
    if ext == ".jsonl":
        return "jsonl"
    elif ext == ".csv":
        return "csv"
    else:
        return "text"
