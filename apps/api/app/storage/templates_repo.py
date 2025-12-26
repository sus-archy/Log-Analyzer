"""
Templates repository - SQL operations for log templates.
"""

from typing import List, Optional, Tuple

import aiosqlite

from ..core.logging import get_logger
from ..core.time import now_utc_iso
from ..schemas.templates import LogTemplate, TemplateWithCount, TemplateOccurrence

logger = get_logger(__name__)


class TemplatesRepo:
    """Repository for log_templates table operations."""
    
    def __init__(self, db: aiosqlite.Connection):
        self.db = db
    
    async def upsert_template(
        self,
        tenant_id: str,
        service_name: str,
        template_hash: int,
        template_text: str,
        timestamp_utc: str,
    ) -> bool:
        """
        Insert or update a template.
        
        Returns:
            True if new template was inserted, False if existing was updated
        """
        now = now_utc_iso()
        
        # Check if template exists first
        cursor = await self.db.execute("""
            SELECT 1 FROM log_templates 
            WHERE tenant_id = ? AND service_name = ? AND template_hash = ?
        """, (tenant_id, service_name, template_hash))
        
        exists = await cursor.fetchone() is not None
        
        if exists:
            # Update existing
            await self.db.execute("""
                UPDATE log_templates SET
                    last_seen_utc = ?
                WHERE tenant_id = ? AND service_name = ? AND template_hash = ?
            """, (now, tenant_id, service_name, template_hash))
            return False
        else:
            # Insert new
            await self.db.execute("""
                INSERT INTO log_templates (
                    tenant_id, service_name, template_hash, template_text,
                    first_seen_utc, last_seen_utc, embedding_state
                )
                VALUES (?, ?, ?, ?, ?, ?, 'queued')
            """, (tenant_id, service_name, template_hash, template_text, timestamp_utc, now))
            return True
    
    async def get_template(
        self,
        tenant_id: str,
        service_name: str,
        template_hash: int,
    ) -> Optional[LogTemplate]:
        """Get a single template by hash. If service_name is empty, finds any matching template."""
        if service_name:
            cursor = await self.db.execute("""
                SELECT tenant_id, service_name, template_hash, template_text,
                       first_seen_utc, last_seen_utc, embedding_state,
                       embedding_model, embedding_updated_utc
                FROM log_templates
                WHERE tenant_id = ? AND service_name = ? AND template_hash = ?
            """, (tenant_id, service_name, template_hash))
        else:
            # Empty service_name means "any service" - find first matching template
            cursor = await self.db.execute("""
                SELECT tenant_id, service_name, template_hash, template_text,
                       first_seen_utc, last_seen_utc, embedding_state,
                       embedding_model, embedding_updated_utc
                FROM log_templates
                WHERE tenant_id = ? AND template_hash = ?
                LIMIT 1
            """, (tenant_id, template_hash))
        
        row = await cursor.fetchone()
        if row is None:
            return None
        
        return LogTemplate(
            tenant_id=row["tenant_id"],
            service_name=row["service_name"],
            template_hash=str(row["template_hash"]),  # Convert to string for JS
            template_text=row["template_text"],
            first_seen_utc=row["first_seen_utc"],
            last_seen_utc=row["last_seen_utc"],
            embedding_state=row["embedding_state"],
            embedding_model=row["embedding_model"],
            embedding_updated_utc=row["embedding_updated_utc"],
        )
    
    async def exists(
        self,
        tenant_id: str,
        service_name: str,
        template_hash: int,
    ) -> bool:
        """Check if template exists."""
        cursor = await self.db.execute("""
            SELECT 1 FROM log_templates
            WHERE tenant_id = ? AND service_name = ? AND template_hash = ?
            LIMIT 1
        """, (tenant_id, service_name, template_hash))
        
        row = await cursor.fetchone()
        return row is not None
    
    async def get_top_templates(
        self,
        tenant_id: str,
        service_name: str,
        from_time: str,
        to_time: str,
        severity_min: int = 4,
        limit: int = 50,
    ) -> List[TemplateWithCount]:
        """Get top templates by occurrence count in time window.
        
        Optimized query that first aggregates counts, then joins for metadata.
        """
        # Step 1: Get top template hashes with counts (fast with index)
        if service_name:
            count_cursor = await self.db.execute("""
                SELECT template_hash, COUNT(*) as count
                FROM logs_stream
                WHERE tenant_id = ?
                  AND service_name = ?
                  AND timestamp_utc >= ?
                  AND timestamp_utc <= ?
                  AND severity >= ?
                GROUP BY template_hash
                ORDER BY count DESC
                LIMIT ?
            """, (tenant_id, service_name, from_time, to_time, severity_min, limit))
        else:
            # No service filter - aggregate across all services
            count_cursor = await self.db.execute("""
                SELECT template_hash, COUNT(*) as count
                FROM logs_stream
                WHERE tenant_id = ?
                  AND timestamp_utc >= ?
                  AND timestamp_utc <= ?
                  AND severity >= ?
                GROUP BY template_hash
                ORDER BY count DESC
                LIMIT ?
            """, (tenant_id, from_time, to_time, severity_min, limit))
        
        count_rows = await count_cursor.fetchall()
        
        if not count_rows:
            return []
        
        # Step 2: Get template metadata for top hashes (fast PK lookup)
        template_hashes = [row["template_hash"] for row in count_rows]
        counts_map = {row["template_hash"]: row["count"] for row in count_rows}
        
        placeholders = ",".join("?" * len(template_hashes))
        meta_cursor = await self.db.execute(f"""
            SELECT template_hash, template_text, first_seen_utc, last_seen_utc
            FROM log_templates
            WHERE tenant_id = ? AND template_hash IN ({placeholders})
        """, [tenant_id] + template_hashes)
        
        meta_rows = await meta_cursor.fetchall()
        meta_map = {row["template_hash"]: row for row in meta_rows}
        
        # Step 3: Build results in count order
        results = []
        for th in template_hashes:
            meta = meta_map.get(th)
            if meta:
                results.append(TemplateWithCount(
                    template_hash=str(th),  # Convert to string for JS
                    template_text=meta["template_text"],
                    count=counts_map[th],
                    sample_log_id=None,  # Skip sample lookup for speed
                    first_seen_utc=meta["first_seen_utc"],
                    last_seen_utc=meta["last_seen_utc"],
                ))
        
        return results
    
    async def get_templates_needing_embedding(
        self,
        limit: int = 100,
    ) -> List[LogTemplate]:
        """Get templates that need embeddings generated."""
        cursor = await self.db.execute("""
            SELECT tenant_id, service_name, template_hash, template_text,
                   first_seen_utc, last_seen_utc, embedding_state,
                   embedding_model, embedding_updated_utc
            FROM log_templates
            WHERE embedding_state IN ('none', 'queued')
            ORDER BY last_seen_utc DESC
            LIMIT ?
        """, (limit,))
        
        rows = await cursor.fetchall()
        return [
            LogTemplate(
                tenant_id=row["tenant_id"],
                service_name=row["service_name"],
                template_hash=str(row["template_hash"]),  # Convert to string for JS
                template_text=row["template_text"],
                first_seen_utc=row["first_seen_utc"],
                last_seen_utc=row["last_seen_utc"],
                embedding_state=row["embedding_state"],
                embedding_model=row["embedding_model"],
                embedding_updated_utc=row["embedding_updated_utc"],
            )
            for row in rows
        ]
    
    async def update_embedding_state(
        self,
        tenant_id: str,
        service_name: str,
        template_hash: int,
        state: str,
        model: str = "",
    ) -> None:
        """Update embedding state for a template."""
        now = now_utc_iso()
        await self.db.execute("""
            UPDATE log_templates
            SET embedding_state = ?,
                embedding_model = ?,
                embedding_updated_utc = ?
            WHERE tenant_id = ? AND service_name = ? AND template_hash = ?
        """, (state, model, now, tenant_id, service_name, template_hash))
    
    async def get_all_services(self, tenant_id: str) -> List[str]:
        """Get all unique service names for a tenant."""
        cursor = await self.db.execute("""
            SELECT DISTINCT service_name
            FROM log_templates
            WHERE tenant_id = ?
            ORDER BY service_name
        """, (tenant_id,))
        
        rows = await cursor.fetchall()
        return [row["service_name"] for row in rows]
    
    async def get_template_count(
        self,
        tenant_id: str,
        service_name: str,
        template_hash: int,
        from_time: str,
        to_time: str,
    ) -> int:
        """Get count of logs matching a template in time window."""
        if service_name:
            cursor = await self.db.execute("""
                SELECT COUNT(*) as count
                FROM logs_stream
                WHERE tenant_id = ?
                  AND service_name = ?
                  AND template_hash = ?
                  AND timestamp_utc >= ?
                  AND timestamp_utc <= ?
            """, (tenant_id, service_name, template_hash, from_time, to_time))
        else:
            # Empty service_name means "any service"
            cursor = await self.db.execute("""
                SELECT COUNT(*) as count
                FROM logs_stream
                WHERE tenant_id = ?
                  AND template_hash = ?
                  AND timestamp_utc >= ?
                  AND timestamp_utc <= ?
            """, (tenant_id, template_hash, from_time, to_time))
        
        row = await cursor.fetchone()
        return row["count"] if row else 0
