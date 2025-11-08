"""Basic tests that don't require model downloads."""

import pytest
import numpy as np
from PIL import Image


def test_imports():
    """Test that all modules can be imported."""
    from vision_rag import (
        download_medmnist,
        load_organmnist_data,
        CLIPImageEncoder,
        ChromaRAGStore,
        ImageSearcher,
    )
    
    # Just verify imports work
    assert download_medmnist is not None
    assert load_organmnist_data is not None
    assert CLIPImageEncoder is not None
    assert ChromaRAGStore is not None
    assert ImageSearcher is not None


def test_get_image_from_array():
    """Test converting numpy array to PIL Image (no downloads needed)."""
    from vision_rag.data_loader import get_image_from_array
    
    # Create a test image array
    image_array = np.random.randint(0, 255, size=(28, 28, 3), dtype=np.uint8)
    
    # Convert to PIL Image
    pil_image = get_image_from_array(image_array)
    
    # Check type and size
    assert isinstance(pil_image, Image.Image)
    assert pil_image.size == (28, 28)
    assert pil_image.mode == "RGB"


def test_chromadb_basic():
    """Test basic ChromaDB operations (no model downloads needed)."""
    import tempfile
    import shutil
    from vision_rag.rag_store import ChromaRAGStore
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create store
        rag_store = ChromaRAGStore(
            collection_name="test_basic",
            persist_directory=temp_dir,
        )
        
        # Verify initialization
        assert rag_store.collection is not None
        assert rag_store.count() == 0
        
        # Add some random embeddings
        embeddings = np.random.randn(5, 512).astype(np.float32)
        rag_store.add_embeddings(embeddings)
        
        # Verify count
        assert rag_store.count() == 5
        
        # Test search
        query = np.random.randn(512).astype(np.float32)
        results = rag_store.search(query, n_results=3)
        
        # Verify results structure
        assert "ids" in results
        assert "distances" in results
        assert "metadatas" in results
        assert len(results["ids"]) == 3
        
    finally:
        shutil.rmtree(temp_dir)
