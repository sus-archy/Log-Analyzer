"""
Vectors repository - SQL operations for template vectors (FAISS mapping).
"""

import base64
from typing import Dict, List, Optional, Tuple

import aiosqlite
import numpy as np

from ..core.logging import get_logger
from ..core.time import now_utc_iso

logger = get_logger(__name__)


class VectorsRepo:
    """Repository for template_vectors table operations."""
    
    def __init__(self, db: aiosqlite.Connection):
        self.db = db
    
    async def upsert_vector(
        self,
        tenant_id: str,
        service_name: str,
        template_hash: int,
        faiss_id: int,
        vector: np.ndarray,
    ) -> None:
        """Insert or update a vector mapping."""
        vector_b64 = base64.b64encode(vector.astype(np.float32).tobytes()).decode("ascii")
        now = now_utc_iso()
        
        await self.db.execute("""
            INSERT INTO template_vectors (
                tenant_id, service_name, template_hash,
                faiss_id, vector_b64, updated_utc
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (tenant_id, service_name, template_hash) DO UPDATE SET
                faiss_id = excluded.faiss_id,
                vector_b64 = excluded.vector_b64,
                updated_utc = excluded.updated_utc
        """, (tenant_id, service_name, template_hash, faiss_id, vector_b64, now))
    
    async def get_mapping_by_faiss_id(
        self,
        faiss_id: int,
    ) -> Optional[Tuple[str, str, int]]:
        """
        Get template info by FAISS ID.
        
        Returns:
            Tuple of (tenant_id, service_name, template_hash) or None
        """
        cursor = await self.db.execute("""
            SELECT tenant_id, service_name, template_hash
            FROM template_vectors
            WHERE faiss_id = ?
        """, (faiss_id,))
        
        row = await cursor.fetchone()
        if row is None:
            return None
        
        return (row["tenant_id"], row["service_name"], row["template_hash"])
    
    async def get_faiss_id(
        self,
        tenant_id: str,
        service_name: str,
        template_hash: int,
    ) -> Optional[int]:
        """Get FAISS ID for a template."""
        cursor = await self.db.execute("""
            SELECT faiss_id
            FROM template_vectors
            WHERE tenant_id = ? AND service_name = ? AND template_hash = ?
        """, (tenant_id, service_name, template_hash))
        
        row = await cursor.fetchone()
        return row["faiss_id"] if row else None
    
    async def get_all_vectors(self) -> List[Tuple[int, str, str, int, np.ndarray]]:
        """
        Get all vectors for rebuilding FAISS index.
        
        Returns:
            List of (faiss_id, tenant_id, service_name, template_hash, vector)
        """
        cursor = await self.db.execute("""
            SELECT faiss_id, tenant_id, service_name, template_hash, vector_b64
            FROM template_vectors
            ORDER BY faiss_id
        """)
        
        rows = await cursor.fetchall()
        results = []
        
        for row in rows:
            try:
                vector_bytes = base64.b64decode(row["vector_b64"])
                vector = np.frombuffer(vector_bytes, dtype=np.float32)
                results.append((
                    row["faiss_id"],
                    row["tenant_id"],
                    row["service_name"],
                    row["template_hash"],
                    vector
                ))
            except Exception as e:
                logger.warning(f"Failed to decode vector for faiss_id={row['faiss_id']}: {e}")
        
        return results
    
    async def get_next_faiss_id(self) -> int:
        """Get next available FAISS ID."""
        cursor = await self.db.execute("""
            SELECT COALESCE(MAX(faiss_id), -1) + 1 as next_id
            FROM template_vectors
        """)
        
        row = await cursor.fetchone()
        return row["next_id"] if row else 0
    
    async def delete_vector(
        self,
        tenant_id: str,
        service_name: str,
        template_hash: int,
    ) -> bool:
        """Delete a vector mapping."""
        cursor = await self.db.execute("""
            DELETE FROM template_vectors
            WHERE tenant_id = ? AND service_name = ? AND template_hash = ?
        """, (tenant_id, service_name, template_hash))
        
        return cursor.rowcount > 0
    
    async def get_vectors_for_templates(
        self,
        template_hashes: List[Tuple[str, str, int]],
    ) -> Dict[Tuple[str, str, int], int]:
        """
        Get FAISS IDs for multiple templates.
        
        Args:
            template_hashes: List of (tenant_id, service_name, template_hash)
            
        Returns:
            Dict mapping (tenant_id, service_name, template_hash) -> faiss_id
        """
        if not template_hashes:
            return {}
        
        # Build query with IN clause
        placeholders = ",".join(["(?, ?, ?)"] * len(template_hashes))
        params = [item for tup in template_hashes for item in tup]
        
        cursor = await self.db.execute(f"""
            SELECT tenant_id, service_name, template_hash, faiss_id
            FROM template_vectors
            WHERE (tenant_id, service_name, template_hash) IN ({placeholders})
        """, params)
        
        rows = await cursor.fetchall()
        return {
            (row["tenant_id"], row["service_name"], row["template_hash"]): row["faiss_id"]
            for row in rows
        }
