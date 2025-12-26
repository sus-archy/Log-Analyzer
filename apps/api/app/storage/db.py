"""
Database connection and initialization - simplified and reliable version.
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

import aiosqlite

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)

# Global connection - keep it simple and reliable
_db_connection: Optional[aiosqlite.Connection] = None
_db_lock = asyncio.Lock()


async def get_db_path() -> Path:
    """Get database path, creating parent directory if needed."""
    db_path = settings.db_path_resolved
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


async def _create_connection() -> aiosqlite.Connection:
    """Create a new optimized database connection."""
    db_path = await get_db_path()
    
    conn = await aiosqlite.connect(
        str(db_path),
        timeout=60.0,  # 60 second timeout
    )
    conn.row_factory = aiosqlite.Row
    
    # Apply performance optimizations - in order of importance
    await conn.execute("PRAGMA busy_timeout=60000")  # 60 second busy timeout FIRST
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA synchronous=NORMAL")
    await conn.execute("PRAGMA cache_size=-200000")  # 200MB cache (negative = KB)
    await conn.execute("PRAGMA temp_store=MEMORY")
    await conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O
    await conn.execute("PRAGMA read_uncommitted=1")
    await conn.execute("PRAGMA wal_autocheckpoint=1000")
    
    return conn


async def init_db() -> aiosqlite.Connection:
    """Initialize database connection and schema."""
    global _db_connection
    
    async with _db_lock:
        if _db_connection is not None:
            # Simple check - try a query
            try:
                await _db_connection.execute("SELECT 1")
                return _db_connection
            except Exception as e:
                logger.warning(f"Connection check failed: {e}, recreating...")
                try:
                    await _db_connection.close()
                except Exception:
                    pass
                _db_connection = None
        
        db_path = await get_db_path()
        logger.info(f"Initializing database at {db_path}")
        
        _db_connection = await _create_connection()
        
        # Run migrations
        from .migrations import run_migrations
        await run_migrations(_db_connection)
        
        logger.info("Database initialized successfully")
        return _db_connection


async def get_db() -> aiosqlite.Connection:
    """Get database connection, initializing if needed."""
    global _db_connection
    
    if _db_connection is None:
        return await init_db()
    
    # Quick health check every call (very fast)
    try:
        await _db_connection.execute("SELECT 1")
        return _db_connection
    except Exception as e:
        logger.warning(f"Database connection lost: {e}")
        _db_connection = None
        return await init_db()


@asynccontextmanager
async def db_transaction() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Context manager for database transactions."""
    db = await get_db()
    try:
        yield db
        await db.commit()
    except Exception:
        await db.rollback()
        raise


async def close_db() -> None:
    """Close database connection."""
    global _db_connection
    
    async with _db_lock:
        if _db_connection is not None:
            try:
                await _db_connection.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            _db_connection = None
            logger.info("Database connection closed")

