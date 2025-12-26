"""Tests for database schema and operations."""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

import aiosqlite

from app.storage.migrations import run_migrations


@pytest.fixture
async def temp_db():
    """Create a temporary database file and connection."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    
    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    await run_migrations(conn)
    
    yield conn
    
    await conn.close()
    os.unlink(path)


@pytest.mark.asyncio
class TestMigrations:
    """Test database migrations."""

    async def test_migrations_create_tables(self, temp_db):
        """Test that migrations create required tables."""
        cursor = await temp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in await cursor.fetchall()}
        
        assert "log_templates" in tables
        assert "logs_stream" in tables
        assert "template_vectors" in tables
        assert "schema_version" in tables

    async def test_migrations_idempotent(self, temp_db):
        """Test that running migrations twice doesn't break things."""
        # Run migrations again
        await run_migrations(temp_db)
        
        # Should still work
        cursor = await temp_db.execute("SELECT MAX(version) FROM schema_version")
        row = await cursor.fetchone()
        assert row[0] == 1


@pytest.mark.asyncio
class TestTemplatesTable:
    """Test log_templates table structure."""

    async def test_insert_template(self, temp_db):
        """Test inserting a template."""
        await temp_db.execute("""
            INSERT INTO log_templates (
                tenant_id, service_name, template_hash, template_text,
                first_seen_utc, last_seen_utc
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("default", "test-service", 12345, "User <*> logged in", 
              "2024-01-15T10:00:00Z", "2024-01-15T10:00:00Z"))
        await temp_db.commit()
        
        cursor = await temp_db.execute(
            "SELECT * FROM log_templates WHERE template_hash = ?", (12345,)
        )
        row = await cursor.fetchone()
        
        assert row is not None
        assert row["template_text"] == "User <*> logged in"

    async def test_primary_key_constraint(self, temp_db):
        """Test that duplicate primary key fails."""
        await temp_db.execute("""
            INSERT INTO log_templates (
                tenant_id, service_name, template_hash, template_text,
                first_seen_utc, last_seen_utc
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("default", "test-service", 12345, "Template 1", 
              "2024-01-15T10:00:00Z", "2024-01-15T10:00:00Z"))
        await temp_db.commit()
        
        with pytest.raises(aiosqlite.IntegrityError):
            await temp_db.execute("""
                INSERT INTO log_templates (
                    tenant_id, service_name, template_hash, template_text,
                    first_seen_utc, last_seen_utc
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("default", "test-service", 12345, "Template 2", 
                  "2024-01-15T10:00:00Z", "2024-01-15T10:00:00Z"))


@pytest.mark.asyncio
class TestLogsStreamTable:
    """Test logs_stream table structure."""

    async def test_insert_log(self, temp_db):
        """Test inserting a log entry."""
        # First insert template
        await temp_db.execute("""
            INSERT INTO log_templates (
                tenant_id, service_name, template_hash, template_text,
                first_seen_utc, last_seen_utc
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("default", "test-service", 12345, "User <*> logged in", 
              "2024-01-15T10:00:00Z", "2024-01-15T10:00:00Z"))
        
        # Now insert log
        cursor = await temp_db.execute("""
            INSERT INTO logs_stream (
                tenant_id, service_name, timestamp_utc, ingest_timestamp_utc,
                severity, template_hash, body_raw
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("default", "test-service", "2024-01-15T10:30:00Z", 
              "2024-01-15T10:30:01Z", 2, 12345, "User alice logged in"))
        await temp_db.commit()
        
        log_id = cursor.lastrowid
        assert log_id is not None
        assert log_id > 0

    async def test_autoincrement_id(self, temp_db):
        """Test that log IDs auto-increment."""
        # Insert template first
        await temp_db.execute("""
            INSERT INTO log_templates (
                tenant_id, service_name, template_hash, template_text,
                first_seen_utc, last_seen_utc
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("default", "test-service", 12345, "Template", 
              "2024-01-15T10:00:00Z", "2024-01-15T10:00:00Z"))
        
        ids = []
        for i in range(3):
            cursor = await temp_db.execute("""
                INSERT INTO logs_stream (
                    tenant_id, service_name, timestamp_utc, ingest_timestamp_utc,
                    severity, template_hash, body_raw
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ("default", "test-service", f"2024-01-15T10:{30+i}:00Z", 
                  f"2024-01-15T10:{30+i}:01Z", 2, 12345, f"Log {i}"))
            ids.append(cursor.lastrowid)
        await temp_db.commit()
        
        # IDs should be incrementing
        assert ids[1] > ids[0]
        assert ids[2] > ids[1]
