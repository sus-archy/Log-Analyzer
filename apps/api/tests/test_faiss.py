"""Tests for FAISS vector index."""

import pytest
import numpy as np
import tempfile
import os

from app.vector.faiss_index import FAISSIndex


class TestFAISSIndex:
    """Test FAISSIndex class."""

    @pytest.fixture
    def index(self):
        """Create a FAISS index for testing."""
        return FAISSIndex(dimension=768)

    @pytest.fixture
    def temp_index_path(self):
        """Create a temporary index file path."""
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_add_single_vector(self, index):
        """Test adding a single vector."""
        vector = np.random.randn(768).astype(np.float32)
        
        await index.add_vector("tenant", "service", 1, vector)
        
        assert index.total == 1

    @pytest.mark.asyncio
    async def test_add_multiple_vectors(self, index):
        """Test adding multiple vectors."""
        for i in range(10):
            vector = np.random.randn(768).astype(np.float32)
            await index.add_vector("tenant", "service", i, vector)
        
        assert index.total == 10

    @pytest.mark.asyncio
    async def test_search_returns_similar_vectors(self, index):
        """Test that search returns similar vectors."""
        base_vector = np.random.randn(768).astype(np.float32)
        
        await index.add_vector("tenant", "service", 1, base_vector)
        
        for i in range(2, 10):
            random_vector = np.random.randn(768).astype(np.float32)
            await index.add_vector("tenant", "service", i, random_vector)
        
        query = base_vector + np.random.randn(768).astype(np.float32) * 0.01
        
        results = await index.search(query, k=5)
        
        assert len(results) > 0
        # First result should have ID 0 (faiss_id, not template_hash)
        assert results[0][0] == 0

    @pytest.mark.asyncio
    async def test_search_with_empty_index(self, index):
        """Test searching an empty index."""
        query = np.random.randn(768).astype(np.float32)
        
        results = await index.search(query, k=5)
        
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_respects_k_limit(self, index):
        """Test that search respects k limit."""
        for i in range(20):
            vector = np.random.randn(768).astype(np.float32)
            await index.add_vector("tenant", "service", i, vector)
        
        query = np.random.randn(768).astype(np.float32)
        
        results = await index.search(query, k=5)
        
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_search_with_mapping(self, index):
        """Test search with template mapping."""
        vector = np.random.randn(768).astype(np.float32)
        await index.add_vector("my-tenant", "my-service", 12345, vector)
        
        results = await index.search_with_mapping(vector, k=1)
        
        assert len(results) == 1
        tenant_id, service_name, template_hash, score = results[0]
        assert tenant_id == "my-tenant"
        assert service_name == "my-service"
        assert template_hash == 12345

    @pytest.mark.asyncio
    async def test_save_and_load(self, index, temp_index_path):
        """Test saving and loading index."""
        for i in range(10):
            vector = np.random.randn(768).astype(np.float32)
            await index.add_vector("tenant", "service", i, vector)
        
        await index.save(temp_index_path)
        
        new_index = FAISSIndex(dimension=768)
        await new_index.load(temp_index_path)
        
        assert new_index.total == index.total

    @pytest.mark.asyncio
    async def test_clear(self, index):
        """Test clearing the index."""
        for i in range(10):
            vector = np.random.randn(768).astype(np.float32)
            await index.add_vector("tenant", "service", i, vector)
        
        assert index.total > 0
        
        await index.clear()
        
        assert index.total == 0

    @pytest.mark.asyncio
    async def test_vector_normalization(self, index):
        """Test that vectors are normalized for cosine similarity."""
        vector = np.random.randn(768).astype(np.float32) * 100
        
        await index.add_vector("tenant", "service", 1, vector)
        
        normalized = vector / np.linalg.norm(vector)
        results = await index.search(normalized, k=1)
        
        assert len(results) > 0
        assert results[0][0] == 0  # faiss_id

    @pytest.mark.asyncio
    async def test_total_property(self, index):
        """Test total property."""
        assert index.total == 0
        
        for i in range(5):
            vector = np.random.randn(768).astype(np.float32)
            await index.add_vector("tenant", "service", i * 10, vector)
        
        assert index.total == 5
