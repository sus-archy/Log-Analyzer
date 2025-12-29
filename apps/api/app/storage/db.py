"""
Database connection pool and initialization.

Provides a connection pool for concurrent database access.
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional, List
from collections import deque

import aiosqlite

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class ConnectionPool:
    """
    Simple async connection pool for aiosqlite.
    
    Manages a pool of database connections for concurrent access.
    """
    
    def __init__(self, db_path: Path, pool_size: int = 5, timeout: float = 30.0):
        self.db_path = db_path
        self.pool_size = pool_size
        self.timeout = timeout
        self._pool: deque[aiosqlite.Connection] = deque()
        self._in_use: set[aiosqlite.Connection] = set()
        self._lock = asyncio.Lock()
        self._initialized = False
        self._closed = False
    
    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new optimized database connection."""
        conn = await aiosqlite.connect(
            str(self.db_path),
            timeout=60.0,
        )
        conn.row_factory = aiosqlite.Row
        
        # Apply performance optimizations
        await conn.execute("PRAGMA busy_timeout=60000")
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA cache_size=-200000")
        await conn.execute("PRAGMA temp_store=MEMORY")
        await conn.execute("PRAGMA mmap_size=268435456")
        await conn.execute("PRAGMA read_uncommitted=1")
        await conn.execute("PRAGMA wal_autocheckpoint=1000")
        
        return conn
    
    async def initialize(self) -> None:
        """Initialize the connection pool with minimum connections."""
        async with self._lock:
            if self._initialized:
                return
            
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create initial connections (half the pool size)
            initial_count = max(1, self.pool_size // 2)
            for _ in range(initial_count):
                try:
                    conn = await self._create_connection()
                    self._pool.append(conn)
                except Exception as e:
                    logger.error(f"Failed to create initial connection: {e}")
            
            self._initialized = True
            logger.info(f"Connection pool initialized with {len(self._pool)} connections")
    
    async def acquire(self) -> aiosqlite.Connection:
        """Acquire a connection from the pool."""
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        start_time = asyncio.get_event_loop().time()
        
        while True:
            async with self._lock:
                # Try to get from pool
                if self._pool:
                    conn = self._pool.popleft()
                    # Verify connection is alive
                    try:
                        await conn.execute("SELECT 1")
                        self._in_use.add(conn)
                        return conn
                    except Exception:
                        # Connection dead, try to close and get another
                        try:
                            await conn.close()
                        except Exception:
                            pass
                        continue
                
                # Create new connection if pool not full
                total_connections = len(self._pool) + len(self._in_use)
                if total_connections < self.pool_size:
                    try:
                        conn = await self._create_connection()
                        self._in_use.add(conn)
                        return conn
                    except Exception as e:
                        logger.error(f"Failed to create connection: {e}")
            
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= self.timeout:
                raise TimeoutError(f"Could not acquire connection within {self.timeout}s")
            
            # Wait and retry
            await asyncio.sleep(0.1)
    
    async def release(self, conn: aiosqlite.Connection) -> None:
        """Release a connection back to the pool."""
        async with self._lock:
            if conn in self._in_use:
                self._in_use.remove(conn)
                if not self._closed:
                    self._pool.append(conn)
                else:
                    try:
                        await conn.close()
                    except Exception:
                        pass
    
    async def close(self) -> None:
        """Close all connections in the pool."""
        async with self._lock:
            self._closed = True
            
            # Close all pooled connections
            while self._pool:
                conn = self._pool.popleft()
                try:
                    await conn.close()
                except Exception as e:
                    logger.warning(f"Error closing pooled connection: {e}")
            
            # Close all in-use connections
            for conn in list(self._in_use):
                try:
                    await conn.close()
                except Exception as e:
                    logger.warning(f"Error closing in-use connection: {e}")
            self._in_use.clear()
            
            logger.info("Connection pool closed")
    
    @property
    def stats(self) -> dict:
        """Get pool statistics."""
        return {
            "pool_size": len(self._pool),
            "in_use": len(self._in_use),
            "max_size": self.pool_size,
            "initialized": self._initialized,
            "closed": self._closed,
        }


# Global connection pool
_pool: Optional[ConnectionPool] = None
_init_lock = asyncio.Lock()


async def get_db_path() -> Path:
    """Get database path, creating parent directory if needed."""
    db_path = settings.db_path_resolved
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


async def init_db() -> aiosqlite.Connection:
    """Initialize database pool and run migrations."""
    global _pool
    
    async with _init_lock:
        if _pool is None:
            db_path = await get_db_path()
            logger.info(f"Initializing database pool at {db_path}")
            
            _pool = ConnectionPool(
                db_path=db_path,
                pool_size=settings.db_pool_size,
                timeout=settings.db_pool_timeout,
            )
            await _pool.initialize()
            
            # Run migrations on a dedicated connection
            conn = await _pool.acquire()
            try:
                from .migrations import run_migrations
                await run_migrations(conn)
                await conn.commit()
            finally:
                await _pool.release(conn)
            
            logger.info("Database initialized successfully")
    
    # Return a connection for compatibility
    return await get_db()


async def get_db() -> aiosqlite.Connection:
    """Get a database connection from the pool."""
    global _pool
    
    if _pool is None:
        await init_db()
    
    return await _pool.acquire()


async def release_db(conn: aiosqlite.Connection) -> None:
    """Release a database connection back to the pool."""
    global _pool
    
    if _pool is not None:
        await _pool.release(conn)


@asynccontextmanager
async def db_connection() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Context manager for database connection with auto-release."""
    conn = await get_db()
    try:
        yield conn
    finally:
        await release_db(conn)


@asynccontextmanager
async def db_transaction() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Context manager for database transactions with commit/rollback."""
    conn = await get_db()
    try:
        yield conn
        await conn.commit()
    except Exception:
        await conn.rollback()
        raise
    finally:
        await release_db(conn)


async def close_db() -> None:
    """Close the database connection pool."""
    global _pool
    
    async with _init_lock:
        if _pool is not None:
            await _pool.close()
            _pool = None
            logger.info("Database pool closed")

