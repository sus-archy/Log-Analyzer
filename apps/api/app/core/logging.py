"""
Logging configuration module.
Sets up structured logging for the application.
"""

import logging
import sys
from typing import Optional

from .config import settings


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """
    Configure application logging.
    
    Args:
        level: Log level override (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured root logger
    """
    log_level = level or settings.log_level
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    return logging.getLogger("logmind")


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"logmind.{name}")


# Initialize logging on import
logger = setup_logging()
