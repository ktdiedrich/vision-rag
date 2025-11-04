"""Tests for ChromaDB RAG store."""

import pytest
import numpy as np
import tempfile
import shutil

from vision_rag.rag_store import ChromaRAGStore


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for ChromaDB."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def rag_store(temp_db_dir):
    """Create a ChromaRAGStore instance."""
    return ChromaRAGStore(
        collection_name="test_collection",
        persist_directory=temp_db_dir,
    )


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings."""
    # Create 10 embeddings with dimension 512
    return np.random.randn(10, 512).astype(np.float32)


def test_rag_store_initialization(rag_store):
    """Test RAG store initialization."""
    assert rag_store.collection_name == "test_collection"
    assert rag_store.collection is not None


def test_add_embeddings(rag_store, sample_embeddings):
    """Test adding embeddings to the store."""
    # ChromaDB requires non-empty metadata, so provide it
    metadatas = [{"index": i} for i in range(len(sample_embeddings))]
    rag_store.add_embeddings(sample_embeddings, metadatas=metadatas)
    
    # Check count
    assert rag_store.count() == len(sample_embeddings)


def test_add_embeddings_with_ids(rag_store, sample_embeddings):
    """Test adding embeddings with custom IDs."""
    ids = [f"custom_id_{i}" for i in range(len(sample_embeddings))]
    rag_store.add_embeddings(sample_embeddings, ids=ids)
    
    assert rag_store.count() == len(sample_embeddings)


def test_add_embeddings_with_metadata(rag_store, sample_embeddings):
    """Test adding embeddings with metadata."""
    metadatas = [{"label": i % 3} for i in range(len(sample_embeddings))]
    rag_store.add_embeddings(sample_embeddings, metadatas=metadatas)
    
    assert rag_store.count() == len(sample_embeddings)


def test_search(rag_store, sample_embeddings):
    """Test searching for similar embeddings."""
    # Add embeddings with metadata
    metadatas = [{"index": i} for i in range(len(sample_embeddings))]
    rag_store.add_embeddings(sample_embeddings, metadatas=metadatas)
    
    # Search with the first embedding
    query_embedding = sample_embeddings[0]
    results = rag_store.search(query_embedding, n_results=3)
    
    # Check results structure
    assert "ids" in results
    assert "distances" in results
    assert "metadatas" in results
    
    # Check number of results
    assert len(results["ids"]) <= 3
    assert len(results["distances"]) == len(results["ids"])
    assert len(results["metadatas"]) == len(results["ids"])


def test_search_returns_closest(rag_store, sample_embeddings):
    """Test that search returns the closest embedding."""
    # Add embeddings with metadata
    metadatas = [{"index": i} for i in range(len(sample_embeddings))]
    rag_store.add_embeddings(sample_embeddings, metadatas=metadatas)
    
    # Search with the first embedding (should find itself first)
    query_embedding = sample_embeddings[0]
    results = rag_store.search(query_embedding, n_results=1)
    
    # The first result should be img_0 (the query itself)
    assert results["ids"][0] == "img_0"
    
    # Distance should be very small (close to 0)
    assert results["distances"][0] < 0.01


def test_count(rag_store, sample_embeddings):
    """Test counting embeddings."""
    assert rag_store.count() == 0
    
    metadatas = [{"index": i} for i in range(len(sample_embeddings))]
    rag_store.add_embeddings(sample_embeddings, metadatas=metadatas)
    assert rag_store.count() == len(sample_embeddings)


def test_clear(rag_store, sample_embeddings):
    """Test clearing the store."""
    # Add embeddings with metadata
    metadatas = [{"index": i} for i in range(len(sample_embeddings))]
    rag_store.add_embeddings(sample_embeddings, metadatas=metadatas)
    assert rag_store.count() > 0
    
    # Clear the store
    rag_store.clear()
    assert rag_store.count() == 0
