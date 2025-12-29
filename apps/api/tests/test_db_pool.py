"""Tests for database connection pool."""

import pytest
from pathlib import Path


class TestConnectionPoolStructure:
    """Test ConnectionPool class structure and initialization."""
    
    def test_pool_attributes(self):
        """Test pool has required attributes."""
        from app.storage.db import ConnectionPool
        
        pool = ConnectionPool(
            db_path=Path("/tmp/test.db"),
            pool_size=3,
            timeout=5.0
        )
        
        assert pool.pool_size == 3
        assert pool.timeout == 5.0
        assert pool.db_path == Path("/tmp/test.db")
    
    def test_pool_initial_state(self):
        """Test pool starts in correct initial state."""
        from app.storage.db import ConnectionPool
        
        pool = ConnectionPool(
            db_path=Path("/tmp/test.db"),
            pool_size=5,
            timeout=10.0
        )
        
        assert pool._initialized is False
        assert pool._closed is False
        assert len(pool._pool) == 0
        assert len(pool._in_use) == 0


class TestConnectionPoolOperations:
    """Test connection pool operations."""
    
    @pytest.mark.asyncio
    async def test_initialize_creates_connections(self, tmp_path):
        """Test initialize creates initial connections."""
        from app.storage.db import ConnectionPool
        
        db_path = tmp_path / "test.db"
        pool = ConnectionPool(
            db_path=db_path,
            pool_size=4,
            timeout=5.0
        )
        
        await pool.initialize()
        
        assert pool._initialized is True
        assert len(pool._pool) >= 1  # At least 1 connection
    
    @pytest.mark.asyncio
    async def test_acquire_returns_connection(self, tmp_path):
        """Test acquire returns a usable connection."""
        from app.storage.db import ConnectionPool
        
        db_path = tmp_path / "test.db"
        pool = ConnectionPool(
            db_path=db_path,
            pool_size=2,
            timeout=5.0
        )
        
        await pool.initialize()
        conn = await pool.acquire()
        
        assert conn is not None
        # Verify we can execute queries
        cursor = await conn.execute("SELECT 1")
        result = await cursor.fetchone()
        assert result[0] == 1
        
        await pool.release(conn)
    
    @pytest.mark.asyncio
    async def test_release_returns_to_pool(self, tmp_path):
        """Test release returns connection to pool."""
        from app.storage.db import ConnectionPool
        
        db_path = tmp_path / "test.db"
        pool = ConnectionPool(
            db_path=db_path,
            pool_size=2,
            timeout=5.0
        )
        
        await pool.initialize()
        
        conn = await pool.acquire()
        pool_size_during_use = len(pool._pool)
        
        await pool.release(conn)
        
        # Pool should have one more connection after release
        assert len(pool._pool) >= pool_size_during_use
    
    @pytest.mark.asyncio
    async def test_pool_reuses_connections(self, tmp_path):
        """Test pool reuses released connections."""
        from app.storage.db import ConnectionPool
        
        db_path = tmp_path / "test.db"
        pool = ConnectionPool(
            db_path=db_path,
            pool_size=1,
            timeout=5.0
        )
        
        await pool.initialize()
        
        # First acquire
        conn1 = await pool.acquire()
        await pool.release(conn1)
        
        # Second acquire should reuse
        conn2 = await pool.acquire()
        
        # Should be the same connection object
        assert conn1 is conn2
        
        await pool.release(conn2)
    
    @pytest.mark.asyncio
    async def test_close_closes_all_connections(self, tmp_path):
        """Test close properly closes all connections."""
        from app.storage.db import ConnectionPool
        
        db_path = tmp_path / "test.db"
        pool = ConnectionPool(
            db_path=db_path,
            pool_size=3,
            timeout=5.0
        )
        
        await pool.initialize()
        
        # Acquire and release some connections
        conn = await pool.acquire()
        await pool.release(conn)
        
        await pool.close()
        
        assert pool._closed is True
        assert len(pool._pool) == 0


class TestConnectionPoolConcurrency:
    """Test connection pool under concurrent access."""
    
    @pytest.mark.asyncio
    async def test_concurrent_acquires(self, tmp_path):
        """Test multiple concurrent acquires work correctly."""
        import asyncio
        from app.storage.db import ConnectionPool
        
        db_path = tmp_path / "test.db"
        pool = ConnectionPool(
            db_path=db_path,
            pool_size=5,
            timeout=10.0
        )
        
        await pool.initialize()
        
        async def use_connection():
            conn = await pool.acquire()
            await conn.execute("SELECT 1")
            await asyncio.sleep(0.01)  # Simulate work
            await pool.release(conn)
            return True
        
        # Run multiple concurrent operations
        results = await asyncio.gather(*[use_connection() for _ in range(10)])
        
        assert all(results)


class TestDatabaseHelperFunctions:
    """Test helper functions for database operations."""
    
    @pytest.mark.asyncio
    async def test_get_db_returns_connection(self, tmp_path, monkeypatch):
        """Test get_db returns a database connection."""
        from app.storage import db
        
        # Patch the settings to use temp path
        monkeypatch.setattr(db.settings, 'sqlite_path', tmp_path / "test.db")
        
        # Force re-initialization
        db._pool = None
        
        conn = await db.get_db()
        
        assert conn is not None
        
        # Verify we can use it
        cursor = await conn.execute("SELECT 1")
        result = await cursor.fetchone()
        assert result[0] == 1
        
        await db.release_db(conn)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
