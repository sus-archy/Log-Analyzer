"""
Semantic search service - handles vector similarity search.
"""

from typing import List, Optional

import numpy as np

from ..core.config import settings
from ..core.logging import get_logger
from ..schemas.chat import SemanticSearchResult, SemanticSearchResponse
from ..storage.db import get_db
from ..storage.templates_repo import TemplatesRepo
from ..storage.logs_repo import LogsRepo
from ..vector.faiss_index import get_faiss_index
from .embedding_service import get_embedding_service

logger = get_logger(__name__)


class SemanticService:
    """Service for semantic search over templates."""
    
    async def search(
        self,
        query: str,
        tenant_id: Optional[str],
        service_name: str,
        from_time: str,
        to_time: str,
        limit: int = 10,
    ) -> SemanticSearchResponse:
        """
        Search templates semantically.
        
        Args:
            query: Natural language query
            tenant_id: Tenant ID
            service_name: Service name filter
            from_time: Start time for counting
            to_time: End time for counting
            limit: Max results
            
        Returns:
            SemanticSearchResponse with matching templates
        """
        if tenant_id is None:
            tenant_id = settings.tenant_id_default
        
        embedding_service = get_embedding_service()
        faiss_index = get_faiss_index()
        
        # Generate query embedding
        query_vector = await embedding_service.embed_text(query)
        
        if query_vector is None:
            logger.warning("Failed to embed query, returning empty results")
            return SemanticSearchResponse(
                results=[],
                query=query,
                service_name=service_name,
                from_time=from_time,
                to_time=to_time,
            )
        
        # Search FAISS
        search_results = await faiss_index.search_with_mapping(
            query_vector,
            k=limit * 2,  # Get extra to filter by service
        )
        
        if not search_results:
            return SemanticSearchResponse(
                results=[],
                query=query,
                service_name=service_name,
                from_time=from_time,
                to_time=to_time,
            )
        
        db = await get_db()
        templates_repo = TemplatesRepo(db)
        logs_repo = LogsRepo(db)
        
        results: List[SemanticSearchResult] = []
        
        for res_tenant_id, res_service_name, template_hash, score in search_results:
            # Filter by tenant
            if res_tenant_id != tenant_id:
                continue
            # Filter by service only if service_name is specified
            if service_name and res_service_name != service_name:
                continue
            
            # Get template
            template = await templates_repo.get_template(
                tenant_id=res_tenant_id,
                service_name=res_service_name,
                template_hash=template_hash,
            )
            
            if template is None:
                continue
            
            # Get count in time window
            count = await templates_repo.get_template_count(
                tenant_id=res_tenant_id,
                service_name=res_service_name,
                template_hash=template_hash,
                from_time=from_time,
                to_time=to_time,
            )
            
            # Get sample log IDs
            sample_logs = await logs_repo.get_sample_logs_for_template(
                tenant_id=res_tenant_id,
                service_name=res_service_name,
                template_hash=template_hash,
                from_time=from_time,
                to_time=to_time,
                limit=3,
            )
            
            results.append(SemanticSearchResult(
                template_hash=str(template_hash),  # Convert to string for JS
                template_text=template.template_text,
                score=score,
                count=count,
                sample_log_ids=[log.id for log in sample_logs],
            ))
            
            if len(results) >= limit:
                break
        
        return SemanticSearchResponse(
            results=results,
            query=query,
            service_name=service_name,
            from_time=from_time,
            to_time=to_time,
        )


# Global service instance
_semantic_service: Optional[SemanticService] = None


def get_semantic_service() -> SemanticService:
    """Get global semantic service instance."""
    global _semantic_service
    if _semantic_service is None:
        _semantic_service = SemanticService()
    return _semantic_service
