"""Tests for image search functionality."""

import pytest
import numpy as np
from PIL import Image
import tempfile
import shutil

from vision_rag.encoder import build_encoder
from vision_rag.rag_store import ChromaRAGStore
from vision_rag.search import ImageSearcher


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for ChromaDB."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def encoder():
    """Create a CLIP encoder instance."""
    return build_encoder(encoder_type="clip", model_name="clip-ViT-B-32")


@pytest.fixture
def rag_store(temp_db_dir):
    """Create a ChromaRAGStore instance."""
    return ChromaRAGStore(
        collection_name="test_search_collection",
        persist_directory=temp_db_dir,
    )


@pytest.fixture
def searcher(encoder, rag_store):
    """Create an ImageSearcher instance."""
    return ImageSearcher(encoder=encoder, rag_store=rag_store)


@pytest.fixture
def sample_images():
    """Create sample images."""
    images = []
    for i in range(5):
        # Create different colored images
        color = (i * 50, (i * 30) % 256, (i * 70) % 256)
        img = Image.new("RGB", (28, 28), color=color)
        images.append(img)
    return images


def test_searcher_initialization(searcher):
    """Test searcher initialization."""
    assert searcher.encoder is not None
    assert searcher.rag_store is not None


def test_search_with_pil_image(searcher, encoder, sample_images):
    """Test searching with a PIL Image."""
    # Add some images to the store first
    embeddings = encoder.encode_images(sample_images)
    searcher.rag_store.add_embeddings(embeddings)
    
    # Search with the first image
    query_image = sample_images[0]
    results = searcher.search(query_image, n_results=3)
    
    # Check results structure
    assert "ids" in results
    assert "distances" in results
    assert "metadatas" in results
    
    # Should find at least one result
    assert len(results["ids"]) > 0


def test_search_with_numpy_image(searcher, encoder, sample_images):
    """Test searching with a numpy array image."""
    # Add some images to the store first
    embeddings = encoder.encode_images(sample_images)
    searcher.rag_store.add_embeddings(embeddings)
    
    # Convert first image to numpy array
    query_image = np.array(sample_images[0])
    results = searcher.search(query_image, n_results=3)
    
    # Check results structure
    assert "ids" in results
    assert len(results["ids"]) > 0


def test_search_finds_similar_image(searcher, encoder, sample_images):
    """Test that search finds the most similar image."""
    # Add images to the store
    embeddings = encoder.encode_images(sample_images)
    searcher.rag_store.add_embeddings(embeddings)
    
    # Search with the first image (should find itself)
    results = searcher.search(sample_images[0], n_results=1)
    
    # The closest match should be img_0 (the query itself)
    assert results["ids"][0] == "img_0"


def test_batch_search(searcher, encoder, sample_images):
    """Test batch searching with multiple query images."""
    # Add images to the store
    embeddings = encoder.encode_images(sample_images)
    searcher.rag_store.add_embeddings(embeddings)
    
    # Search with multiple queries
    query_images = sample_images[:2]
    results_list = searcher.batch_search(query_images, n_results=2)
    
    # Check that we get results for each query
    assert len(results_list) == len(query_images)
    
    # Each result should have the expected structure
    for results in results_list:
        assert "ids" in results
        assert "distances" in results
        assert "metadatas" in results


def test_search_with_different_n_results(searcher, encoder, sample_images):
    """Test searching with different n_results values."""
    # Add images to the store
    embeddings = encoder.encode_images(sample_images)
    searcher.rag_store.add_embeddings(embeddings)
    
    # Search with n_results=1
    results_1 = searcher.search(sample_images[0], n_results=1)
    assert len(results_1["ids"]) == 1
    
    # Search with n_results=3
    results_3 = searcher.search(sample_images[0], n_results=3)
    assert len(results_3["ids"]) <= 3


class FakeEncoder:
    """Simple fake encoder that returns the numpy array passed as the embedding.

    This allows tests to construct embeddings directly and use them as both
    stored embeddings and query embeddings.
    """
    def __init__(self, embedding_dim: int = 2):
        self._dim = embedding_dim

    def encode_image(self, image):
        # If image is numpy array matching embedding dim, return as-is
        if isinstance(image, np.ndarray):
            return image.astype(float)
        # If PIL Image, convert to numpy array - but tests pass numpy vectors
        raise ValueError("FakeEncoder expects numpy array as query")

    def encode_images(self, images):
        return np.array(images, dtype=float)

    @property
    def embedding_dimension(self):
        return self._dim


def test_classify_knn(temp_db_dir):
    """Test the k-NN classify method returns the majority label."""
    # Create a rag store and add embeddings with labels
    rag = ChromaRAGStore(collection_name="test_knn_classify", persist_directory=temp_db_dir)
    # Two embeddings near (1,0) with label 0, two near (10,10) with label 1
    embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [10.0, 10.0],
        [10.1, 10.0],
    ], dtype=float)

    metadatas = [
        {"label": 0},
        {"label": 0},
        {"label": 1},
        {"label": 1},
    ]
    rag.add_embeddings(embeddings, metadatas=metadatas)

    fake_encoder = FakeEncoder(embedding_dim=2)
    searcher = ImageSearcher(encoder=fake_encoder, rag_store=rag)

    # Query near the first cluster
    query = np.array([1.05, -0.05])
    result = searcher.classify(query, n_results=3)
    assert result["label"] == 0
    assert result["count"] >= 2
    assert result["confidence"] >= 2/3 - 1e-6


def test_batch_classify(temp_db_dir):
    """Test batch classification returns expected labels for multiple queries."""
    rag = ChromaRAGStore(collection_name="test_batch_knn", persist_directory=temp_db_dir)
    embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [10.0, 10.0],
        [10.1, 10.0],
    ], dtype=float)
    metadatas = [
        {"label": 0},
        {"label": 0},
        {"label": 1},
        {"label": 1},
    ]
    rag.add_embeddings(embeddings, metadatas=metadatas)

    fake_encoder = FakeEncoder(embedding_dim=2)
    searcher = ImageSearcher(encoder=fake_encoder, rag_store=rag)

    queries = [np.array([1.05, -0.05]), np.array([10.05, 10.02])]
    results = searcher.batch_classify(queries, n_results=3)
    assert isinstance(results, list) and len(results) == 2
    assert results[0]["label"] == 0
    assert results[1]["label"] == 1
