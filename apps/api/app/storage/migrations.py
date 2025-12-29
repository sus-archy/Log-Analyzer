"""
Database migrations.
Creates and updates SQLite schema.
"""

import aiosqlite

from ..core.logging import get_logger

logger = get_logger(__name__)


# Schema version for tracking migrations
SCHEMA_VERSION = 2


async def run_migrations(db: aiosqlite.Connection) -> None:
    """Run all database migrations."""
    logger.info("Running database migrations...")
    
    # Create schema version table
    await db.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    
    # Check current version
    cursor = await db.execute("SELECT MAX(version) FROM schema_version")
    row = await cursor.fetchone()
    current_version = row[0] if row and row[0] else 0
    
    if current_version < 1:
        await migrate_v1(db)
        await db.execute("INSERT INTO schema_version (version) VALUES (?)", (1,))
    
    if current_version < 2:
        await migrate_v2(db)
        await db.execute("INSERT INTO schema_version (version) VALUES (?)", (2,))
    
    await db.commit()
    logger.info(f"Database schema at version {SCHEMA_VERSION}")


async def migrate_v1(db: aiosqlite.Connection) -> None:
    """Initial schema creation."""
    logger.info("Applying migration v1: Initial schema")
    
    # log_templates table
    await db.execute("""
        CREATE TABLE IF NOT EXISTS log_templates (
            tenant_id TEXT NOT NULL,
            service_name TEXT NOT NULL,
            
            template_hash INTEGER NOT NULL,
            template_text TEXT NOT NULL,
            
            first_seen_utc TEXT NOT NULL,
            last_seen_utc TEXT NOT NULL,
            
            embedding_state TEXT NOT NULL DEFAULT 'none',
            embedding_model TEXT NOT NULL DEFAULT '',
            embedding_updated_utc TEXT NOT NULL DEFAULT '',
            
            PRIMARY KEY (tenant_id, service_name, template_hash)
        )
    """)
    
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_templates_service
        ON log_templates (tenant_id, service_name)
    """)
    
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_templates_embedding_state
        ON log_templates (embedding_state)
    """)
    
    # logs_stream table
    await db.execute("""
        CREATE TABLE IF NOT EXISTS logs_stream (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            
            tenant_id TEXT NOT NULL,
            service_name TEXT NOT NULL,
            environment TEXT NOT NULL DEFAULT 'prod',
            
            timestamp_utc TEXT NOT NULL,
            ingest_timestamp_utc TEXT NOT NULL,
            
            severity INTEGER NOT NULL,
            
            host TEXT NOT NULL DEFAULT '',
            
            template_hash INTEGER NOT NULL,
            parameters_json TEXT NOT NULL DEFAULT '[]',
            
            trace_id TEXT NOT NULL DEFAULT '',
            span_id TEXT NOT NULL DEFAULT '',
            
            attributes_json TEXT NOT NULL DEFAULT '{}',
            
            body_raw TEXT NOT NULL DEFAULT ''
        )
    """)
    
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_logs_main
        ON logs_stream (tenant_id, service_name, timestamp_utc, severity, template_hash)
    """)
    
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_logs_trace
        ON logs_stream (tenant_id, service_name, trace_id)
    """)
    
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_logs_template
        ON logs_stream (tenant_id, service_name, template_hash)
    """)
    
    # template_vectors table for FAISS mapping
    await db.execute("""
        CREATE TABLE IF NOT EXISTS template_vectors (
            tenant_id TEXT NOT NULL,
            service_name TEXT NOT NULL,
            template_hash INTEGER NOT NULL,
            
            faiss_id INTEGER NOT NULL,
            vector_b64 TEXT NOT NULL,
            updated_utc TEXT NOT NULL,
            
            PRIMARY KEY (tenant_id, service_name, template_hash)
        )
    """)
    
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_vectors_faiss_id
        ON template_vectors (faiss_id)
    """)
    
    # ingested_files table for tracking processed files and detecting duplicates
    await db.execute("""
        CREATE TABLE IF NOT EXISTS ingested_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            
            file_path TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            file_hash TEXT NOT NULL,
            
            lines_processed INTEGER NOT NULL DEFAULT 0,
            events_inserted INTEGER NOT NULL DEFAULT 0,
            templates_discovered INTEGER NOT NULL DEFAULT 0,
            
            ingested_at TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            
            UNIQUE(file_hash)
        )
    """)
    
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_ingested_files_hash
        ON ingested_files (file_hash)
    """)
    
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_ingested_files_path
        ON ingested_files (file_path)
    """)
    
    logger.info("Migration v1 complete")


async def migrate_v2(db: aiosqlite.Connection) -> None:
    """Add performance indexes and analyze tables."""
    logger.info("Applying migration v2: Performance indexes")
    
    # Add timestamp-only index for time range queries
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_logs_timestamp
        ON logs_stream (timestamp_utc)
    """)
    
    # Add severity index for filtering
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_logs_severity
        ON logs_stream (severity)
    """)
    
    # Add tenant + timestamp index for common queries
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_logs_tenant_time
        ON logs_stream (tenant_id, timestamp_utc)
    """)
    
    # Add index for service listing
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_logs_service
        ON logs_stream (service_name)
    """)
    
    # Analyze tables to update query planner statistics
    await db.execute("ANALYZE logs_stream")
    await db.execute("ANALYZE log_templates")
    await db.execute("ANALYZE template_vectors")
    
    logger.info("Migration v2 complete")
