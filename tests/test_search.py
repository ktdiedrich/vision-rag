"""Tests for image search functionality."""

import pytest
import numpy as np
from PIL import Image
import tempfile
import shutil

from vision_rag.encoder import CLIPImageEncoder
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
    return CLIPImageEncoder(model_name="clip-ViT-B-32")


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
