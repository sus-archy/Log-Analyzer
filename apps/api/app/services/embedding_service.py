"""
Embedding service - generates and manages embeddings for templates.
"""

import asyncio
from typing import List, Optional, Tuple

import numpy as np

from ..core.config import settings
from ..core.logging import get_logger
from ..llm.ollama_client import get_ollama_client, OllamaError
from ..storage.db import get_db
from ..storage.templates_repo import TemplatesRepo
from ..storage.vectors_repo import VectorsRepo
from ..vector.faiss_index import get_faiss_index
from ..vector.vector_codec import list_to_vector

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating and managing template embeddings."""
    
    def __init__(self):
        self._processing = False
        self._lock = asyncio.Lock()
    
    async def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None on failure
        """
        ollama = get_ollama_client()
        
        try:
            embedding_list = await ollama.embed(text)
            return list_to_vector(embedding_list)
        except OllamaError as e:
            logger.error(f"Embedding failed: {e}")
            return None
    
    async def process_pending_templates(
        self,
        batch_size: int = 50,
        max_templates: int = 100,
    ) -> int:
        """
        Process templates that need embeddings.
        
        Args:
            batch_size: Number of templates to process per batch
            max_templates: Maximum templates to process in this call
            
        Returns:
            Number of templates processed
        """
        async with self._lock:
            if self._processing:
                logger.debug("Already processing embeddings")
                return 0
            self._processing = True
        
        try:
            db = await get_db()
            templates_repo = TemplatesRepo(db)
            vectors_repo = VectorsRepo(db)
            faiss_index = get_faiss_index()
            
            # Get templates needing embeddings
            templates = await templates_repo.get_templates_needing_embedding(max_templates)
            
            if not templates:
                return 0
            
            logger.info(f"Processing {len(templates)} templates for embedding")
            
            processed = 0
            for template in templates:
                try:
                    # Generate embedding
                    vector = await self.embed_text(template.template_text)
                    
                    if vector is None:
                        await templates_repo.update_embedding_state(
                            tenant_id=template.tenant_id,
                            service_name=template.service_name,
                            template_hash=template.template_hash,
                            state="failed",
                        )
                        continue
                    
                    # Add to FAISS index
                    faiss_id = await faiss_index.add_vector(
                        tenant_id=template.tenant_id,
                        service_name=template.service_name,
                        template_hash=template.template_hash,
                        vector=vector,
                    )
                    
                    # Store in database
                    await vectors_repo.upsert_vector(
                        tenant_id=template.tenant_id,
                        service_name=template.service_name,
                        template_hash=template.template_hash,
                        faiss_id=faiss_id,
                        vector=vector,
                    )
                    
                    # Update template state
                    await templates_repo.update_embedding_state(
                        tenant_id=template.tenant_id,
                        service_name=template.service_name,
                        template_hash=template.template_hash,
                        state="ready",
                        model=settings.ollama_embed_model,
                    )
                    
                    processed += 1
                    
                except Exception as e:
                    logger.error(f"Failed to embed template {template.template_hash}: {e}")
                    await templates_repo.update_embedding_state(
                        tenant_id=template.tenant_id,
                        service_name=template.service_name,
                        template_hash=template.template_hash,
                        state="failed",
                    )
            
            await db.commit()
            
            # Save FAISS index
            if processed > 0:
                await faiss_index.save()
            
            logger.info(f"Embedded {processed} templates")
            return processed
            
        finally:
            self._processing = False
    
    async def rebuild_index_from_db(self) -> int:
        """
        Rebuild FAISS index from stored vectors.
        
        Returns:
            Number of vectors loaded
        """
        db = await get_db()
        vectors_repo = VectorsRepo(db)
        faiss_index = get_faiss_index()
        
        # Get all vectors from database
        vectors_data = await vectors_repo.get_all_vectors()
        
        if not vectors_data:
            logger.info("No vectors in database to rebuild from")
            return 0
        
        # Rebuild index
        await faiss_index.rebuild_from_vectors(vectors_data)
        
        logger.info(f"Rebuilt index with {len(vectors_data)} vectors")
        return len(vectors_data)
    
    async def ensure_index_loaded(self) -> None:
        """
        Ensure FAISS index is loaded or rebuilt.
        
        Called on startup to initialize the index.
        """
        faiss_index = get_faiss_index()
        
        # Try loading from file
        loaded = await faiss_index.load()
        
        if loaded:
            # Index loaded, need to restore mappings from DB
            db = await get_db()
            vectors_repo = VectorsRepo(db)
            vectors_data = await vectors_repo.get_all_vectors()
            
            for faiss_id, tenant_id, service_name, template_hash, _ in vectors_data:
                faiss_index.set_mapping(faiss_id, tenant_id, service_name, template_hash)
            
            logger.info(f"Restored {len(vectors_data)} mappings for FAISS index")
        else:
            # Try to rebuild from database
            count = await self.rebuild_index_from_db()
            if count == 0:
                logger.info("No existing vectors, starting with empty index")

    async def get_embedding_stats(self) -> dict:
        """
        Get embedding generation statistics.
        
        Returns:
            Dict with total_templates, embedded_count, pending_count, percentage
        """
        db = await get_db()
        
        # Get total templates count
        cursor = await db.execute("SELECT COUNT(*) FROM log_templates")
        row = await cursor.fetchone()
        total_templates = row[0] if row else 0
        
        # Get embedded count
        cursor = await db.execute(
            "SELECT COUNT(*) FROM log_templates WHERE embedding_state = 'ready'"
        )
        row = await cursor.fetchone()
        embedded_count = row[0] if row else 0
        
        # Get pending count
        cursor = await db.execute(
            "SELECT COUNT(*) FROM log_templates WHERE embedding_state IN ('none', 'queued')"
        )
        row = await cursor.fetchone()
        pending_count = row[0] if row else 0
        
        # Get failed count
        cursor = await db.execute(
            "SELECT COUNT(*) FROM log_templates WHERE embedding_state = 'failed'"
        )
        row = await cursor.fetchone()
        failed_count = row[0] if row else 0
        
        percentage = (embedded_count / total_templates * 100) if total_templates > 0 else 0
        
        return {
            "total_templates": total_templates,
            "embedded_count": embedded_count,
            "pending_count": pending_count,
            "failed_count": failed_count,
            "percentage": round(percentage, 1),
        }


# Global service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
