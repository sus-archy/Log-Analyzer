"""
Query service - handles log and template queries.
"""

from typing import List, Optional

from ..core.config import settings
from ..core.logging import get_logger
from ..schemas.logs import LogEventResponse, LogQueryResponse
from ..schemas.templates import (
    TemplateWithCount,
    TopTemplatesResponse,
    TemplateDetailResponse,
    TemplateOccurrence,
)
from ..storage.db import db_connection
from ..storage.templates_repo import TemplatesRepo
from ..storage.logs_repo import LogsRepo

logger = get_logger(__name__)


class QueryService:
    """Service for querying logs and templates."""
    
    async def query_logs(
        self,
        tenant_id: Optional[str],
        from_time: str,
        to_time: str,
        service_name: Optional[str] = None,
        severity_min: Optional[int] = None,
        template_hash: Optional[int] = None,
        trace_id: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> LogQueryResponse:
        """
        Query logs with filters.
        
        Args:
            tenant_id: Tenant ID (uses default if None)
            from_time: Start time (ISO 8601)
            to_time: End time (ISO 8601)
            service_name: Filter by service
            severity_min: Minimum severity level
            template_hash: Filter by template
            trace_id: Filter by trace
            limit: Max results
            offset: Pagination offset
            
        Returns:
            LogQueryResponse with logs and pagination info
        """
        if tenant_id is None:
            tenant_id = settings.tenant_id_default
        
        async with db_connection() as db:
            logs_repo = LogsRepo(db)
            
            logs, total = await logs_repo.query_logs(
                tenant_id=tenant_id,
                from_time=from_time,
                to_time=to_time,
                service_name=service_name,
                severity_min=severity_min,
                template_hash=template_hash,
                trace_id=trace_id,
                limit=limit,
                offset=offset,
            )
            
            return LogQueryResponse(
                logs=logs,
                total=total,
                limit=limit,
                offset=offset,
                has_more=(offset + len(logs)) < total,
            )
    
    async def get_top_templates(
        self,
        tenant_id: Optional[str],
        service_name: str,
        from_time: str,
        to_time: str,
        severity_min: int = 4,
        limit: int = 50,
    ) -> TopTemplatesResponse:
        """
        Get top templates by occurrence count.
        
        Args:
            tenant_id: Tenant ID
            service_name: Service name (required)
            from_time: Start time
            to_time: End time
            severity_min: Minimum severity (default ERROR=4)
            limit: Max templates to return
            
        Returns:
            TopTemplatesResponse with ranked templates
        """
        if tenant_id is None:
            tenant_id = settings.tenant_id_default
        
        async with db_connection() as db:
            templates_repo = TemplatesRepo(db)
            
            templates = await templates_repo.get_top_templates(
                tenant_id=tenant_id,
                service_name=service_name,
                from_time=from_time,
                to_time=to_time,
                severity_min=severity_min,
                limit=limit,
            )
            
            return TopTemplatesResponse(
                templates=templates,
                service_name=service_name,
                from_time=from_time,
                to_time=to_time,
                total=len(templates),
            )
    
    async def get_template_detail(
        self,
        tenant_id: Optional[str],
        service_name: str,
        template_hash: int,
        from_time: str,
        to_time: str,
        occurrences_limit: int = 20,
    ) -> Optional[TemplateDetailResponse]:
        """
        Get detailed template information with occurrences.
        
        Args:
            tenant_id: Tenant ID
            service_name: Service name
            template_hash: Template hash
            from_time: Start time
            to_time: End time
            occurrences_limit: Max occurrences to include
            
        Returns:
            TemplateDetailResponse or None if not found
        """
        if tenant_id is None:
            tenant_id = settings.tenant_id_default
        
        async with db_connection() as db:
            templates_repo = TemplatesRepo(db)
            logs_repo = LogsRepo(db)
            
            # Get template metadata
            template = await templates_repo.get_template(
                tenant_id=tenant_id,
                service_name=service_name,
                template_hash=template_hash,
            )
            
            if template is None:
                return None
            
            # Get count in window
            total_count = await templates_repo.get_template_count(
                tenant_id=tenant_id,
                service_name=service_name,
                template_hash=template_hash,
                from_time=from_time,
                to_time=to_time,
            )
            
            # Get sample logs
            sample_logs = await logs_repo.get_sample_logs_for_template(
                tenant_id=tenant_id,
                service_name=service_name,
                template_hash=template_hash,
                from_time=from_time,
                to_time=to_time,
                limit=occurrences_limit,
            )
            
            occurrences = [
                TemplateOccurrence(
                    log_id=log.id,
                    timestamp_utc=log.timestamp_utc,
                    parameters=log.parameters,
                    severity=log.severity,
                    host=log.host,
                )
                for log in sample_logs
            ]
            
            return TemplateDetailResponse(
                template_hash=template.template_hash,
                template_text=template.template_text,
                service_name=template.service_name,
                first_seen_utc=template.first_seen_utc,
                last_seen_utc=template.last_seen_utc,
                total_count=total_count,
                occurrences=occurrences,
                embedding_state=template.embedding_state,
            )
    
    async def get_services(
        self,
        tenant_id: Optional[str] = None,
    ) -> List[str]:
        """
        Get list of all service names.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            List of service names
        """
        if tenant_id is None:
            tenant_id = settings.tenant_id_default
        
        async with db_connection() as db:
            logs_repo = LogsRepo(db)
            return await logs_repo.get_services(tenant_id)
    
    async def get_log_by_id(
        self,
        tenant_id: Optional[str],
        log_id: int,
    ) -> Optional[LogEventResponse]:
        """
        Get a single log by ID.
        
        Args:
            tenant_id: Tenant ID
            log_id: Log ID
            
        Returns:
            LogEventResponse or None
        """
        if tenant_id is None:
            tenant_id = settings.tenant_id_default
        
        async with db_connection() as db:
            logs_repo = LogsRepo(db)
            return await logs_repo.get_log_by_id(tenant_id, log_id)
    
    async def get_quick_stats(
        self,
        tenant_id: Optional[str] = None,
    ) -> dict:
        """
        Get quick database statistics using SQLite internal stats.
        This is much faster than COUNT(*) on large tables.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            Dictionary with log count, template count, service count
        """
        if tenant_id is None:
            tenant_id = settings.tenant_id_default
        
        async with db_connection() as db:
            # Use max(id) as a fast approximation of count (assumes auto-increment IDs)
            # This is O(1) instead of O(n) for COUNT(*)
            try:
                # Fast log count estimate using max ID
                cursor = await db.execute("SELECT MAX(id), MIN(id) FROM logs_stream")
                row = await cursor.fetchone()
                max_id, min_id = row[0] or 0, row[1] or 0
                estimated_logs = max_id - min_id + 1 if max_id else 0
                
                # Template count (usually smaller, direct count is OK)
                cursor = await db.execute("SELECT COUNT(*) FROM log_templates")
                row = await cursor.fetchone()
                template_count = row[0] if row else 0
                
                # Service count from distinct values (smaller result set)
                cursor = await db.execute("SELECT COUNT(DISTINCT service_name) FROM log_templates")
                row = await cursor.fetchone()
                service_count = row[0] if row else 0
                
                return {
                    "logs_estimated": estimated_logs,
                    "templates": template_count,
                    "services": service_count,
                    "is_estimated": True,
                }
            except Exception as e:
                logger.error(f"Quick stats failed: {e}")
                return {
                    "logs_estimated": 0,
                    "templates": 0,
                    "services": 0,
                    "is_estimated": True,
                    "error": str(e),
                }


# Global service instance
_query_service: Optional[QueryService] = None


def get_query_service() -> QueryService:
    """Get global query service instance."""
    global _query_service
    if _query_service is None:
        _query_service = QueryService()
    return _query_service
