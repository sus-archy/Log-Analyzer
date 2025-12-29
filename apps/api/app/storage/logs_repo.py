"""
Logs repository - SQL operations for log events.
"""

import json
from typing import List, Optional, Tuple

import aiosqlite

from ..core.logging import get_logger
from ..schemas.logs import NormalizedLogEvent, LogEventResponse, severity_name

logger = get_logger(__name__)


class LogsRepo:
    """Repository for logs_stream table operations."""
    
    def __init__(self, db: aiosqlite.Connection):
        self.db = db
    
    async def insert_log(self, event: NormalizedLogEvent) -> int:
        """
        Insert a single log event.
        
        Returns:
            The inserted log ID
        """
        cursor = await self.db.execute("""
            INSERT INTO logs_stream (
                tenant_id, service_name, environment,
                timestamp_utc, ingest_timestamp_utc, severity,
                host, template_hash, parameters_json,
                trace_id, span_id, attributes_json, body_raw
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.tenant_id,
            event.service_name,
            event.environment,
            event.timestamp_utc,
            event.ingest_timestamp_utc,
            event.severity,
            event.host,
            event.template_hash,
            event.parameters_json,
            event.trace_id,
            event.span_id,
            event.attributes_json,
            event.body_raw,
        ))
        
        return cursor.lastrowid or 0
    
    async def insert_logs_batch(self, events: List[NormalizedLogEvent]) -> int:
        """
        Insert multiple log events in a batch.
        
        Returns:
            Number of events inserted
        """
        if not events:
            return 0
        
        await self.db.executemany("""
            INSERT INTO logs_stream (
                tenant_id, service_name, environment,
                timestamp_utc, ingest_timestamp_utc, severity,
                host, template_hash, parameters_json,
                trace_id, span_id, attributes_json, body_raw
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (
                e.tenant_id, e.service_name, e.environment,
                e.timestamp_utc, e.ingest_timestamp_utc, e.severity,
                e.host, e.template_hash, e.parameters_json,
                e.trace_id, e.span_id, e.attributes_json, e.body_raw,
            )
            for e in events
        ])
        
        return len(events)
    
    async def query_logs(
        self,
        tenant_id: str,
        from_time: str,
        to_time: str,
        service_name: Optional[str] = None,
        severity_min: Optional[int] = None,
        template_hash: Optional[int] = None,
        trace_id: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> Tuple[List[LogEventResponse], int]:
        """
        Query logs with filters.
        Optimized: Uses LIMIT-based count estimation for large datasets.
        
        Returns:
            Tuple of (logs, total_count)
        """
        # Build query dynamically - order conditions for optimal index usage
        # Service name first (if present) for idx_logs_service_time index
        conditions = []
        params: list[str | int] = []
        
        if service_name:
            conditions.append("l.service_name = ?")
            params.append(service_name)
        
        # Then timestamp range
        conditions.extend(["l.timestamp_utc >= ?", "l.timestamp_utc <= ?"])
        params.extend([from_time, to_time])
        
        # Tenant last (less selective)
        conditions.append("l.tenant_id = ?")
        params.append(tenant_id)
        
        if severity_min is not None:
            conditions.append("l.severity >= ?")
            params.append(severity_min)
        
        if template_hash is not None:
            conditions.append("l.template_hash = ?")
            params.append(template_hash)
        
        if trace_id:
            conditions.append("l.trace_id = ?")
            params.append(trace_id)
        
        where_clause = " AND ".join(conditions)
        
        # Optimized: Use index hint and fetch logs first, then join
        # Determine which index to use based on query pattern
        index_hint = ""
        if service_name:
            # Service-specific query - use composite index
            index_hint = "INDEXED BY idx_logs_service_time"
        
        # Get logs first (fast with LIMIT and proper index)
        cursor = await self.db.execute(f"""
            SELECT l.*, t.template_text
            FROM logs_stream l {index_hint}
            LEFT JOIN log_templates t ON 
                l.tenant_id = t.tenant_id AND
                l.service_name = t.service_name AND
                l.template_hash = t.template_hash
            WHERE {where_clause}
            ORDER BY l.timestamp_utc DESC
            LIMIT ? OFFSET ?
        """, params + [limit, offset])
        
        rows = await cursor.fetchall()
        logs = []
        
        for row in rows:
            try:
                parameters = json.loads(row["parameters_json"]) if row["parameters_json"] else []
            except json.JSONDecodeError:
                parameters = []
            
            try:
                attributes = json.loads(row["attributes_json"]) if row["attributes_json"] else {}
            except json.JSONDecodeError:
                attributes = {}
            
            logs.append(LogEventResponse(
                id=row["id"],
                tenant_id=row["tenant_id"],
                service_name=row["service_name"],
                environment=row["environment"],
                timestamp_utc=row["timestamp_utc"],
                severity=row["severity"],
                severity_name=severity_name(row["severity"]),
                host=row["host"],
                template_hash=row["template_hash"],
                template_text=row["template_text"],
                parameters=parameters,
                trace_id=row["trace_id"],
                span_id=row["span_id"],
                attributes=attributes,
                body_raw=row["body_raw"],
            ))
        
        # Get total count - skip expensive COUNT(*) for better performance
        # We determine "has_more" based on whether we got a full page
        if len(logs) < limit:
            # Got fewer than requested = we're at the end
            total = offset + len(logs)
        else:
            # Got a full page, check if there's one more row beyond current page
            # This is much faster than COUNT(*) on millions of rows
            try:
                check_cursor = await self.db.execute(
                    f"""SELECT 1 FROM logs_stream l {index_hint}
                        WHERE {where_clause}
                        LIMIT 1 OFFSET ?""",
                    params + [offset + limit]
                )
                has_more = await check_cursor.fetchone()
                if has_more:
                    # There's more data - use a placeholder total that indicates this
                    # Frontend will show "Loading more..." pagination
                    total = offset + limit + 1  # Signal there's at least one more page
                else:
                    total = offset + len(logs)
            except Exception:
                # If check fails, assume there's more
                total = offset + limit + 1
        
        return logs, total
    
    async def get_log_by_id(
        self,
        tenant_id: str,
        log_id: int,
    ) -> Optional[LogEventResponse]:
        """Get a single log by ID."""
        cursor = await self.db.execute("""
            SELECT l.*, t.template_text
            FROM logs_stream l
            LEFT JOIN log_templates t ON 
                l.tenant_id = t.tenant_id AND
                l.service_name = t.service_name AND
                l.template_hash = t.template_hash
            WHERE l.tenant_id = ? AND l.id = ?
        """, (tenant_id, log_id))
        
        row = await cursor.fetchone()
        if row is None:
            return None
        
        try:
            parameters = json.loads(row["parameters_json"]) if row["parameters_json"] else []
        except json.JSONDecodeError:
            parameters = []
        
        try:
            attributes = json.loads(row["attributes_json"]) if row["attributes_json"] else {}
        except json.JSONDecodeError:
            attributes = {}
        
        return LogEventResponse(
            id=row["id"],
            tenant_id=row["tenant_id"],
            service_name=row["service_name"],
            environment=row["environment"],
            timestamp_utc=row["timestamp_utc"],
            severity=row["severity"],
            severity_name=severity_name(row["severity"]),
            host=row["host"],
            template_hash=row["template_hash"],
            template_text=row["template_text"],
            parameters=parameters,
            trace_id=row["trace_id"],
            span_id=row["span_id"],
            attributes=attributes,
            body_raw=row["body_raw"],
        )
    
    async def get_sample_logs_for_template(
        self,
        tenant_id: str,
        service_name: str,
        template_hash: int,
        from_time: str,
        to_time: str,
        limit: int = 10,
    ) -> List[LogEventResponse]:
        """Get sample logs for a template. If service_name is empty, searches all services."""
        if service_name:
            cursor = await self.db.execute("""
                SELECT l.*, t.template_text
                FROM logs_stream l
                LEFT JOIN log_templates t ON 
                    l.tenant_id = t.tenant_id AND
                    l.service_name = t.service_name AND
                    l.template_hash = t.template_hash
                WHERE l.tenant_id = ?
                  AND l.service_name = ?
                  AND l.template_hash = ?
                  AND l.timestamp_utc >= ?
                  AND l.timestamp_utc <= ?
                ORDER BY l.timestamp_utc DESC
                LIMIT ?
            """, (tenant_id, service_name, template_hash, from_time, to_time, limit))
        else:
            # Empty service_name means "any service"
            cursor = await self.db.execute("""
                SELECT l.*, t.template_text
                FROM logs_stream l
                LEFT JOIN log_templates t ON 
                    l.tenant_id = t.tenant_id AND
                    l.service_name = t.service_name AND
                    l.template_hash = t.template_hash
                WHERE l.tenant_id = ?
                  AND l.template_hash = ?
                  AND l.timestamp_utc >= ?
                  AND l.timestamp_utc <= ?
                ORDER BY l.timestamp_utc DESC
                LIMIT ?
            """, (tenant_id, template_hash, from_time, to_time, limit))
        
        rows = await cursor.fetchall()
        logs = []
        
        for row in rows:
            try:
                parameters = json.loads(row["parameters_json"]) if row["parameters_json"] else []
            except json.JSONDecodeError:
                parameters = []
            
            try:
                attributes = json.loads(row["attributes_json"]) if row["attributes_json"] else {}
            except json.JSONDecodeError:
                attributes = {}
            
            logs.append(LogEventResponse(
                id=row["id"],
                tenant_id=row["tenant_id"],
                service_name=row["service_name"],
                environment=row["environment"],
                timestamp_utc=row["timestamp_utc"],
                severity=row["severity"],
                severity_name=severity_name(row["severity"]),
                host=row["host"],
                template_hash=row["template_hash"],
                template_text=row["template_text"],
                parameters=parameters,
                trace_id=row["trace_id"],
                span_id=row["span_id"],
                attributes=attributes,
                body_raw=row["body_raw"],
            ))
        
        return logs
    
    async def get_services(self, tenant_id: str) -> List[str]:
        """Get all unique service names."""
        cursor = await self.db.execute("""
            SELECT DISTINCT service_name
            FROM logs_stream
            WHERE tenant_id = ?
            ORDER BY service_name
        """, (tenant_id,))
        
        rows = await cursor.fetchall()
        return [row["service_name"] for row in rows]
