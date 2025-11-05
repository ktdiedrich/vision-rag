"""Tests for visualization module."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from PIL import Image

from vision_rag.visualization import RAGVisualizer


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for visualization outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_images():
    """Create sample images for testing."""
    # Create some sample grayscale images
    images = []
    for i in range(10):
        img_array = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        images.append(img_array)
    return images


@pytest.fixture
def sample_labels():
    """Create sample labels for testing."""
    return [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    return np.random.randn(10, 512)


def test_visualizer_initialization(temp_output_dir):
    """Test RAGVisualizer initialization."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    assert visualizer.output_dir == Path(temp_output_dir)
    assert visualizer.output_dir.exists()


def test_save_input_images_grid(temp_output_dir, sample_images, sample_labels):
    """Test saving input images grid."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    output_path = visualizer.save_input_images_grid(
        images=sample_images[:8],  # Use 8 images for a 4x2 grid
        labels=sample_labels[:8],
        filename="test_input_grid.png",
        grid_size=(2, 4)
    )
    
    assert Path(output_path).exists()
    assert "test_input_grid.png" in output_path


def test_save_search_input_images(temp_output_dir, sample_images, sample_labels):
    """Test saving search input images."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    output_path = visualizer.save_search_input_images(
        images=sample_images[:5],
        labels=sample_labels[:5],
        filename="test_search_input.png"
    )
    
    assert Path(output_path).exists()
    assert "test_search_input.png" in output_path


def test_save_search_results(temp_output_dir, sample_images, sample_labels):
    """Test saving search results."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    query_image = sample_images[0]
    query_label = sample_labels[0]
    retrieved_images = sample_images[1:4]
    retrieved_metadata = [{"label": label} for label in sample_labels[1:4]]
    distances = [0.1, 0.3, 0.5]
    
    output_path = visualizer.save_search_results(
        query_image=query_image,
        query_label=query_label,
        retrieved_images=retrieved_images,
        retrieved_metadata=retrieved_metadata,
        distances=distances,
        filename="test_search_results.png"
    )
    
    assert Path(output_path).exists()
    assert "test_search_results.png" in output_path


def test_save_label_distribution(temp_output_dir, sample_labels):
    """Test saving label distribution."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    output_path = visualizer.save_label_distribution(
        labels=sample_labels,
        filename="test_label_dist.png"
    )
    
    assert Path(output_path).exists()
    assert "test_label_dist.png" in output_path


def test_save_embedding_space_visualization(temp_output_dir, sample_embeddings, sample_labels):
    """Test saving embedding space visualization."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    # Test with PCA (should always be available)
    output_path = visualizer.save_embedding_space_visualization(
        embeddings=sample_embeddings,
        labels=sample_labels,
        method='pca',
        filename="test_embedding_space.png"
    )
    
    assert Path(output_path).exists()
    assert "test_embedding_space.png" in output_path


def test_save_embedding_space_visualization_tsne(temp_output_dir, sample_embeddings, sample_labels):
    """Test saving embedding space visualization with t-SNE."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    # Test with t-SNE
    output_path = visualizer.save_embedding_space_visualization(
        embeddings=sample_embeddings,
        labels=sample_labels,
        method='tsne',
        filename="test_embedding_tsne.png"
    )
    
    assert Path(output_path).exists()
    assert "test_embedding_tsne.png" in output_path


def test_pil_image_support(temp_output_dir):
    """Test that visualizer works with PIL Images."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    # Create PIL Images
    pil_images = []
    for i in range(5):
        img_array = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        pil_img = Image.fromarray(img_array, mode='L')
        pil_images.append(pil_img)
    
    output_path = visualizer.save_input_images_grid(
        images=pil_images,
        labels=[0, 1, 2, 0, 1],
        filename="test_pil_images.png",
        grid_size=(1, 5)
    )
    
    assert Path(output_path).exists()


def test_rgb_image_support(temp_output_dir):
    """Test that visualizer works with RGB images."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    # Create RGB images
    rgb_images = []
    for i in range(3):
        img_array = np.random.randint(0, 255, (28, 28, 3), dtype=np.uint8)
        rgb_images.append(img_array)
    
    output_path = visualizer.save_input_images_grid(
        images=rgb_images,
        labels=[0, 1, 2],
        filename="test_rgb_images.png",
        grid_size=(1, 3)
    )
    
    assert Path(output_path).exists()


def test_invalid_embedding_method(temp_output_dir, sample_embeddings, sample_labels):
    """Test that invalid embedding methods raise ValueError."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    with pytest.raises(ValueError, match="Unknown method"):
        visualizer.save_embedding_space_visualization(
            embeddings=sample_embeddings,
            labels=sample_labels,
            method='invalid_method'
        )