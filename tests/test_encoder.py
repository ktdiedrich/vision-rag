"""Tests for CLIP image encoder."""

import pytest
import numpy as np
from PIL import Image

from vision_rag.encoder import CLIPImageEncoder


@pytest.fixture
def encoder():
    """Create a CLIP encoder instance."""
    return CLIPImageEncoder(model_name="clip-ViT-B-32")


@pytest.fixture
def sample_image():
    """Create a sample image."""
    return Image.new("RGB", (28, 28), color=(100, 150, 200))


@pytest.fixture
def sample_image_array():
    """Create a sample image as numpy array."""
    return np.random.randint(0, 255, size=(28, 28, 3), dtype=np.uint8)


def test_encoder_initialization(encoder):
    """Test encoder initialization."""
    assert encoder.model_name == "clip-ViT-B-32"
    assert encoder.model is not None


def test_encode_pil_image(encoder, sample_image):
    """Test encoding a PIL Image."""
    embedding = encoder.encode_image(sample_image)
    
    # Check that embedding is a numpy array
    assert isinstance(embedding, np.ndarray)
    
    # Check embedding dimension
    assert len(embedding.shape) == 1
    assert embedding.shape[0] > 0


def test_encode_numpy_image(encoder, sample_image_array):
    """Test encoding a numpy array image."""
    embedding = encoder.encode_image(sample_image_array)
    
    # Check that embedding is a numpy array
    assert isinstance(embedding, np.ndarray)
    
    # Check embedding dimension
    assert len(embedding.shape) == 1
    assert embedding.shape[0] > 0


def test_encode_multiple_images(encoder, sample_image, sample_image_array):
    """Test encoding multiple images."""
    images = [sample_image, sample_image_array, sample_image]
    
    embeddings = encoder.encode_images(images)
    
    # Check shape
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] > 0
    
    # Check that embeddings are different for different images
    assert not np.allclose(embeddings[0], embeddings[1])


def test_embedding_dimension(encoder):
    """Test getting embedding dimension."""
    dim = encoder.embedding_dimension
    assert isinstance(dim, int)
    assert dim > 0


def test_consistent_encoding(encoder, sample_image):
    """Test that encoding the same image produces consistent results."""
    embedding1 = encoder.encode_image(sample_image)
    embedding2 = encoder.encode_image(sample_image)
    
    # Embeddings should be very similar (allowing for minor floating point differences)
    assert np.allclose(embedding1, embedding2, rtol=1e-5)
