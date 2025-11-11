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


def test_save_input_images_without_labels(temp_output_dir, sample_images):
    """Test saving input images grid without labels."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    output_path = visualizer.save_input_images_grid(
        images=sample_images[:6],
        labels=None,
        filename="test_no_labels.png",
        grid_size=(2, 3)
    )
    
    assert Path(output_path).exists()


def test_save_input_images_max_images(temp_output_dir, sample_images, sample_labels):
    """Test that max_images parameter limits the number of displayed images."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    output_path = visualizer.save_input_images_grid(
        images=sample_images,
        labels=sample_labels,
        filename="test_max_images.png",
        max_images=5,
        grid_size=(2, 3)
    )
    
    assert Path(output_path).exists()


def test_save_input_images_custom_title(temp_output_dir, sample_images, sample_labels):
    """Test saving input images with custom title."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    output_path = visualizer.save_input_images_grid(
        images=sample_images[:4],
        labels=sample_labels[:4],
        filename="test_custom_title.png",
        title="Custom Test Title",
        grid_size=(2, 2)
    )
    
    assert Path(output_path).exists()


def test_save_search_input_without_labels(temp_output_dir, sample_images):
    """Test saving search input images without labels."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    output_path = visualizer.save_search_input_images(
        images=sample_images[:3],
        labels=None,
        filename="test_search_no_labels.png"
    )
    
    assert Path(output_path).exists()


def test_save_search_input_max_images(temp_output_dir, sample_images, sample_labels):
    """Test max_images parameter in search input images."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    # Provide more images than max_images
    output_path = visualizer.save_search_input_images(
        images=sample_images,
        labels=sample_labels,
        filename="test_search_max.png",
        max_images=3
    )
    
    assert Path(output_path).exists()


def test_save_search_input_single_image(temp_output_dir, sample_images, sample_labels):
    """Test search input with single image."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    output_path = visualizer.save_search_input_images(
        images=[sample_images[0]],
        labels=[sample_labels[0]],
        filename="test_search_single.png"
    )
    
    assert Path(output_path).exists()


def test_save_search_results_without_query_label(temp_output_dir, sample_images, sample_labels):
    """Test search results without query label."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    query_image = sample_images[0]
    retrieved_images = sample_images[1:3]
    retrieved_metadata = [{"label": label} for label in sample_labels[1:3]]
    distances = [0.2, 0.4]
    
    output_path = visualizer.save_search_results(
        query_image=query_image,
        query_label=None,
        retrieved_images=retrieved_images,
        retrieved_metadata=retrieved_metadata,
        distances=distances,
        filename="test_search_no_query_label.png"
    )
    
    assert Path(output_path).exists()


def test_save_search_results_single_result(temp_output_dir, sample_images, sample_labels):
    """Test search results with single retrieved image."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    output_path = visualizer.save_search_results(
        query_image=sample_images[0],
        query_label=sample_labels[0],
        retrieved_images=[sample_images[1]],
        retrieved_metadata=[{"label": sample_labels[1]}],
        distances=[0.15],
        filename="test_search_single_result.png"
    )
    
    assert Path(output_path).exists()


def test_save_search_results_no_label_in_metadata(temp_output_dir, sample_images, sample_labels):
    """Test search results when metadata doesn't contain label."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    output_path = visualizer.save_search_results(
        query_image=sample_images[0],
        query_label=sample_labels[0],
        retrieved_images=sample_images[1:3],
        retrieved_metadata=[{"other_field": "value"}, {"other_field": "value2"}],
        distances=[0.1, 0.2],
        filename="test_search_no_metadata_label.png"
    )
    
    assert Path(output_path).exists()


def test_save_search_results_rgb_images(temp_output_dir, sample_labels):
    """Test search results with RGB images."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    # Create RGB images
    query_image = np.random.randint(0, 255, (28, 28, 3), dtype=np.uint8)
    retrieved_images = [
        np.random.randint(0, 255, (28, 28, 3), dtype=np.uint8) for _ in range(2)
    ]
    
    output_path = visualizer.save_search_results(
        query_image=query_image,
        query_label=sample_labels[0],
        retrieved_images=retrieved_images,
        retrieved_metadata=[{"label": label} for label in sample_labels[1:3]],
        distances=[0.1, 0.3],
        filename="test_search_rgb.png"
    )
    
    assert Path(output_path).exists()


def test_save_search_results_pil_images(temp_output_dir, sample_labels):
    """Test search results with PIL images."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    # Create PIL images
    query_array = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    query_image = Image.fromarray(query_array, mode='L')
    
    retrieved_images = []
    for _ in range(2):
        img_array = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        retrieved_images.append(Image.fromarray(img_array, mode='L'))
    
    output_path = visualizer.save_search_results(
        query_image=query_image,
        query_label=sample_labels[0],
        retrieved_images=retrieved_images,
        retrieved_metadata=[{"label": label} for label in sample_labels[1:3]],
        distances=[0.2, 0.5],
        filename="test_search_pil.png"
    )
    
    assert Path(output_path).exists()


def test_save_label_distribution_custom_title(temp_output_dir, sample_labels):
    """Test label distribution with custom title."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    output_path = visualizer.save_label_distribution(
        labels=sample_labels,
        filename="test_label_custom_title.png",
        title="Custom Label Distribution"
    )
    
    assert Path(output_path).exists()


def test_save_label_distribution_many_classes(temp_output_dir):
    """Test label distribution with many different classes."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    # Create labels with many classes
    labels = list(range(11)) * 5  # 11 different classes
    
    output_path = visualizer.save_label_distribution(
        labels=labels,
        filename="test_label_many_classes.png"
    )
    
    assert Path(output_path).exists()


def test_save_embedding_space_umap(temp_output_dir, sample_embeddings, sample_labels):
    """Test embedding space visualization with UMAP."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    # Test with UMAP (will fall back to PCA if umap not installed)
    output_path = visualizer.save_embedding_space_visualization(
        embeddings=sample_embeddings,
        labels=sample_labels,
        method='umap',
        filename="test_embedding_umap.png"
    )
    
    assert Path(output_path).exists()


def test_save_embedding_space_custom_title(temp_output_dir, sample_embeddings, sample_labels):
    """Test embedding space with custom title."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    output_path = visualizer.save_embedding_space_visualization(
        embeddings=sample_embeddings,
        labels=sample_labels,
        method='pca',
        filename="test_embedding_custom.png",
        title="Custom Embedding Title"
    )
    
    assert Path(output_path).exists()


def test_save_embedding_space_small_dataset(temp_output_dir):
    """Test embedding space with very small dataset."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    # Create a very small dataset
    small_embeddings = np.random.randn(6, 128)
    small_labels = [0, 1, 0, 1, 2, 2]
    
    output_path = visualizer.save_embedding_space_visualization(
        embeddings=small_embeddings,
        labels=small_labels,
        method='tsne',
        filename="test_embedding_small.png"
    )
    
    assert Path(output_path).exists()


def test_save_embedding_space_many_labels(temp_output_dir):
    """Test embedding space with more than MAX_COLORBAR_LABELS labels."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    # Create dataset with 15 different labels (> MAX_COLORBAR_LABELS)
    embeddings = np.random.randn(60, 256)
    labels = list(range(15)) * 4
    
    output_path = visualizer.save_embedding_space_visualization(
        embeddings=embeddings,
        labels=labels,
        method='pca',
        filename="test_embedding_many_labels.png"
    )
    
    assert Path(output_path).exists()


def test_visualizer_creates_output_directory(temp_output_dir):
    """Test that visualizer creates nested output directory if it doesn't exist."""
    nested_dir = Path(temp_output_dir) / "nested" / "directory"
    visualizer = RAGVisualizer(output_dir=str(nested_dir))
    
    assert nested_dir.exists()
    assert visualizer.output_dir == nested_dir


def test_save_search_input_different_grid_sizes(temp_output_dir, sample_images):
    """Test search input images automatically adjusts grid size."""
    visualizer = RAGVisualizer(output_dir=temp_output_dir)
    
    # Test with 6 images (should create 2x3 or 3x2 grid)
    output_path = visualizer.save_search_input_images(
        images=sample_images[:6],
        filename="test_search_grid_6.png"
    )
    assert Path(output_path).exists()
    
    # Test with 7 images
    output_path = visualizer.save_search_input_images(
        images=sample_images[:7],
        filename="test_search_grid_7.png"
    )
    assert Path(output_path).exists()