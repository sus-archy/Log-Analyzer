"""
Ingested files repository - tracks which files have been processed.
"""

import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

import aiosqlite

from ..core.logging import get_logger
from ..core.time import now_utc_iso

logger = get_logger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """
    Compute a SHA-256 hash of file contents.
    Uses chunked reading for large files.
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


class IngestedFilesRepo:
    """Repository for tracking ingested files."""
    
    def __init__(self, db: aiosqlite.Connection):
        self.db = db
    
    async def is_file_ingested(self, file_hash: str) -> bool:
        """
        Check if a file with this hash has already been ingested.
        
        Args:
            file_hash: SHA-256 hash of the file
            
        Returns:
            True if file was already ingested
        """
        cursor = await self.db.execute(
            "SELECT id FROM ingested_files WHERE file_hash = ?",
            (file_hash,)
        )
        row = await cursor.fetchone()
        return row is not None
    
    async def get_ingested_file(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get details of an ingested file by its hash.
        
        Args:
            file_hash: SHA-256 hash of the file
            
        Returns:
            File record or None
        """
        cursor = await self.db.execute("""
            SELECT id, file_path, file_name, file_size, file_hash,
                   lines_processed, events_inserted, templates_discovered,
                   ingested_at, tenant_id
            FROM ingested_files
            WHERE file_hash = ?
        """, (file_hash,))
        row = await cursor.fetchone()
        
        if row:
            return {
                "id": row[0],
                "file_path": row[1],
                "file_name": row[2],
                "file_size": row[3],
                "file_hash": row[4],
                "lines_processed": row[5],
                "events_inserted": row[6],
                "templates_discovered": row[7],
                "ingested_at": row[8],
                "tenant_id": row[9],
            }
        return None
    
    async def record_ingested_file(
        self,
        file_path: Path,
        file_hash: str,
        lines_processed: int,
        events_inserted: int,
        templates_discovered: int,
        tenant_id: str,
    ) -> int:
        """
        Record that a file has been ingested.
        
        Args:
            file_path: Path to the file
            file_hash: SHA-256 hash of the file
            lines_processed: Number of lines processed
            events_inserted: Number of events inserted
            templates_discovered: Number of templates discovered
            tenant_id: Tenant ID
            
        Returns:
            ID of the record
        """
        file_size = file_path.stat().st_size if file_path.exists() else 0
        
        cursor = await self.db.execute("""
            INSERT INTO ingested_files (
                file_path, file_name, file_size, file_hash,
                lines_processed, events_inserted, templates_discovered,
                ingested_at, tenant_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(file_path),
            file_path.name,
            file_size,
            file_hash,
            lines_processed,
            events_inserted,
            templates_discovered,
            now_utc_iso(),
            tenant_id,
        ))
        
        return cursor.lastrowid or 0
    
    async def get_all_ingested_files(self, tenant_id: Optional[str] = None) -> list:
        """
        Get all ingested files, optionally filtered by tenant.
        
        Args:
            tenant_id: Optional tenant filter
            
        Returns:
            List of ingested file records
        """
        if tenant_id:
            cursor = await self.db.execute("""
                SELECT id, file_path, file_name, file_size, file_hash,
                       lines_processed, events_inserted, templates_discovered,
                       ingested_at, tenant_id
                FROM ingested_files
                WHERE tenant_id = ?
                ORDER BY ingested_at DESC
            """, (tenant_id,))
        else:
            cursor = await self.db.execute("""
                SELECT id, file_path, file_name, file_size, file_hash,
                       lines_processed, events_inserted, templates_discovered,
                       ingested_at, tenant_id
                FROM ingested_files
                ORDER BY ingested_at DESC
            """)
        
        rows = await cursor.fetchall()
        return [
            {
                "id": row[0],
                "file_path": row[1],
                "file_name": row[2],
                "file_size": row[3],
                "file_hash": row[4],
                "lines_processed": row[5],
                "events_inserted": row[6],
                "templates_discovered": row[7],
                "ingested_at": row[8],
                "tenant_id": row[9],
            }
            for row in rows
        ]
    
    async def clear_all(self) -> int:
        """
        Clear all ingested file records.
        
        Returns:
            Number of records deleted
        """
        cursor = await self.db.execute("SELECT COUNT(*) FROM ingested_files")
        row = await cursor.fetchone()
        count = row[0] if row else 0
        
        await self.db.execute("DELETE FROM ingested_files")
        
        return count