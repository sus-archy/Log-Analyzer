"""
FAISS index - manages vector similarity search.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from ..core.config import settings
from ..core.logging import get_logger
from .vector_codec import normalize_vector, normalize_vectors, list_to_vector

logger = get_logger(__name__)


class FAISSIndex:
    """
    FAISS vector index for template semantic search.
    
    Uses IndexFlatIP (inner product) with normalized vectors
    for cosine similarity.
    """
    
    def __init__(self, dimension: int = 768):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Vector dimension (depends on embedding model)
        """
        self.dimension = dimension
        self.index: Optional[faiss.IndexFlatIP] = None
        self._id_counter = 0
        self._lock = asyncio.Lock()
        
        # Mapping from FAISS ID to (tenant_id, service_name, template_hash)
        self._id_to_template: Dict[int, Tuple[str, str, int]] = {}
    
    def _ensure_index(self) -> faiss.IndexFlatIP:
        """Ensure index is initialized."""
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
        return self.index
    
    async def add_vector(
        self,
        tenant_id: str,
        service_name: str,
        template_hash: int,
        vector: np.ndarray,
    ) -> int:
        """
        Add a vector to the index.
        
        Args:
            tenant_id: Tenant ID
            service_name: Service name
            template_hash: Template hash
            vector: Embedding vector
            
        Returns:
            FAISS ID assigned to the vector
        """
        async with self._lock:
            index = self._ensure_index()
            
            # Check dimension
            if len(vector) != self.dimension:
                # Update dimension on first vector
                if index.ntotal == 0:
                    self.dimension = len(vector)
                    self.index = faiss.IndexFlatIP(self.dimension)
                    index = self.index
                else:
                    raise ValueError(
                        f"Vector dimension {len(vector)} doesn't match index {self.dimension}"
                    )
            
            # Normalize for cosine similarity
            normalized = normalize_vector(vector).reshape(1, -1).astype(np.float32)
            
            # Get ID
            faiss_id = self._id_counter
            self._id_counter += 1
            
            # Add to index
            index.add(normalized)  # type: ignore[call-arg]
            
            # Store mapping
            self._id_to_template[faiss_id] = (tenant_id, service_name, template_hash)
            
            return faiss_id
    
    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding
            k: Number of results
            
        Returns:
            List of (faiss_id, score) tuples
        """
        async with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            
            # Normalize query
            normalized = normalize_vector(query_vector).reshape(1, -1).astype(np.float32)
            
            # Search
            index = self.index  # Already checked above
            k = min(k, index.ntotal)
            distances, indices = index.search(normalized, k)  # type: ignore[call-arg]
            
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0:  # FAISS returns -1 for no match
                    results.append((int(idx), float(dist)))
            
            return results
    
    async def search_with_mapping(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[str, str, int, float]]:
        """
        Search and return template info.
        
        Args:
            query_vector: Query embedding
            k: Number of results
            
        Returns:
            List of (tenant_id, service_name, template_hash, score) tuples
        """
        results = await self.search(query_vector, k)
        
        mapped = []
        for faiss_id, score in results:
            if faiss_id in self._id_to_template:
                tenant_id, service_name, template_hash = self._id_to_template[faiss_id]
                mapped.append((tenant_id, service_name, template_hash, score))
        
        return mapped
    
    async def get_template_for_id(
        self,
        faiss_id: int,
    ) -> Optional[Tuple[str, str, int]]:
        """Get template info for a FAISS ID."""
        return self._id_to_template.get(faiss_id)
    
    async def save(self, path: Optional[Path] = None) -> None:
        """
        Save index to disk.
        
        Args:
            path: Path to save to (defaults to config)
        """
        if path is None:
            path = settings.faiss_index_path_resolved
        elif isinstance(path, str):
            path = Path(path)
        
        async with self._lock:
            if self.index is None or self.index.ntotal == 0:
                logger.info("No vectors to save")
                return
            
            path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(path))
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors to {path}")
    
    async def load(self, path: Optional[Path] = None) -> bool:
        """
        Load index from disk.
        
        Args:
            path: Path to load from (defaults to config)
            
        Returns:
            True if loaded successfully
        """
        if path is None:
            path = settings.faiss_index_path_resolved
        elif isinstance(path, str):
            path = Path(path)
        
        async with self._lock:
            if not path.exists():
                logger.info(f"No FAISS index file at {path}")
                return False
            
            try:
                loaded_index = faiss.read_index(str(path))
                self.index = loaded_index
                self.dimension = loaded_index.d
                logger.info(f"Loaded FAISS index with {loaded_index.ntotal} vectors from {path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                return False
    
    async def rebuild_from_vectors(
        self,
        vectors_data: List[Tuple[int, str, str, int, np.ndarray]],
    ) -> None:
        """
        Rebuild index from stored vectors.
        
        Args:
            vectors_data: List of (faiss_id, tenant_id, service_name, template_hash, vector)
        """
        async with self._lock:
            if not vectors_data:
                logger.info("No vectors to rebuild from")
                return
            
            # Get dimension from first vector
            first_vec = vectors_data[0][4]
            self.dimension = len(first_vec)
            
            # Create new index
            self.index = faiss.IndexFlatIP(self.dimension)
            self._id_to_template = {}
            self._id_counter = 0
            
            # Collect vectors and mappings
            all_vectors = []
            for faiss_id, tenant_id, service_name, template_hash, vector in vectors_data:
                normalized = normalize_vector(vector)
                all_vectors.append(normalized)
                self._id_to_template[faiss_id] = (tenant_id, service_name, template_hash)
                self._id_counter = max(self._id_counter, faiss_id + 1)
            
            # Add all at once
            if all_vectors and self.index is not None:
                vectors_array = np.vstack(all_vectors).astype(np.float32)
                self.index.add(vectors_array)  # type: ignore[call-arg]
            
            logger.info(f"Rebuilt FAISS index with {len(vectors_data)} vectors")
    
    @property
    def size(self) -> int:
        """Get number of vectors in index."""
        if self.index is None:
            return 0
        return self.index.ntotal
    
    @property
    def total(self) -> int:
        """Get number of vectors in index (alias for size)."""
        return self.size
    
    async def clear(self) -> None:
        """Clear the index and reset all state."""
        async with self._lock:
            self.index = faiss.IndexFlatIP(self.dimension)
            self._id_to_template = {}
            self._id_counter = 0
            logger.info("Cleared FAISS index")
    
    def set_mapping(self, faiss_id: int, tenant_id: str, service_name: str, template_hash: int) -> None:
        """Set mapping for a FAISS ID (used during rebuild)."""
        self._id_to_template[faiss_id] = (tenant_id, service_name, template_hash)
        self._id_counter = max(self._id_counter, faiss_id + 1)


# Global index instance
_faiss_index: Optional[FAISSIndex] = None


def get_faiss_index() -> FAISSIndex:
    """Get global FAISS index instance."""
    global _faiss_index
    if _faiss_index is None:
        _faiss_index = FAISSIndex()
    return _faiss_index


async def init_faiss_index() -> FAISSIndex:
    """Initialize and load FAISS index."""
    index = get_faiss_index()
    await index.load()
    return index
