"""Integration tests for the complete vision-RAG pipeline."""

import pytest
import tempfile
import shutil

from vision_rag import (
    download_organmnist,
    load_organmnist_data,
    CLIPImageEncoder,
    ChromaRAGStore,
    ImageSearcher,
)


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for ChromaDB."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_end_to_end_pipeline(temp_data_dir, temp_db_dir):
    """Test the complete vision-RAG pipeline end-to-end."""
    
    # Step 1: Download OrganSMNIST dataset
    download_organmnist(root=temp_data_dir)
    
    # Step 2: Load training data (use a small subset for testing)
    train_images, train_labels = load_organmnist_data(split="train", root=temp_data_dir)
    
    # Use only first 20 images for faster testing
    train_images_subset = train_images[:20]
    train_labels_subset = train_labels[:20]
    
    # Step 3: Initialize encoder
    encoder = CLIPImageEncoder(model_name="clip-ViT-B-32")
    
    # Step 4: Encode training images
    train_embeddings = encoder.encode_images(
        [img for img in train_images_subset]
    )
    
    # Check embeddings shape
    assert train_embeddings.shape[0] == len(train_images_subset)
    assert train_embeddings.shape[1] == encoder.embedding_dimension
    
    # Step 5: Create RAG store and add embeddings
    rag_store = ChromaRAGStore(
        collection_name="organmnist_test",
        persist_directory=temp_db_dir,
    )
    
    # Add embeddings with metadata
    metadatas = [{"label": int(label)} for label in train_labels_subset]
    rag_store.add_embeddings(train_embeddings, metadatas=metadatas)
    
    # Verify embeddings were added
    assert rag_store.count() == len(train_images_subset)
    
    # Step 6: Load test data
    test_images, test_labels = load_organmnist_data(split="test", root=temp_data_dir)
    
    # Use a small subset for testing
    test_images_subset = test_images[:5]
    
    # Step 7: Create searcher and perform searches
    searcher = ImageSearcher(encoder=encoder, rag_store=rag_store)
    
    # Search with a test image
    results = searcher.search(test_images_subset[0], n_results=3)
    
    # Verify results
    assert len(results["ids"]) <= 3
    assert len(results["distances"]) == len(results["ids"])
    assert len(results["metadatas"]) == len(results["ids"])
    
    # Verify metadata contains labels
    for metadata in results["metadatas"]:
        assert "label" in metadata
        assert isinstance(metadata["label"], int)
    
    # Step 8: Test batch search
    batch_results = searcher.batch_search(
        [img for img in test_images_subset], 
        n_results=3
    )
    
    # Verify batch results
    assert len(batch_results) == len(test_images_subset)
    
    for result in batch_results:
        assert "ids" in result
        assert len(result["ids"]) <= 3


def test_pipeline_with_same_image(temp_data_dir, temp_db_dir):
    """Test that searching with a training image returns itself as the top result."""
    
    # Download and load data
    download_organmnist(root=temp_data_dir)
    train_images, train_labels = load_organmnist_data(split="train", root=temp_data_dir)
    
    # Use small subset
    train_images_subset = train_images[:10]
    
    # Encode and store
    encoder = CLIPImageEncoder(model_name="clip-ViT-B-32")
    embeddings = encoder.encode_images([img for img in train_images_subset])
    
    rag_store = ChromaRAGStore(
        collection_name="organmnist_same_image_test",
        persist_directory=temp_db_dir,
    )
    rag_store.add_embeddings(embeddings)
    
    # Search with the same image that was stored
    searcher = ImageSearcher(encoder=encoder, rag_store=rag_store)
    results = searcher.search(train_images_subset[0], n_results=1)
    
    # The top result should be img_0 (the query itself)
    assert results["ids"][0] == "img_0"
    
    # Distance should be very small
    assert results["distances"][0] < 0.01
