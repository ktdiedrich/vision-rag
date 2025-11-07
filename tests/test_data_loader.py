"""Tests for data loading module."""

import pytest
import numpy as np
from PIL import Image
import tempfile
import shutil
from pathlib import Path

from vision_rag.data_loader import (
    download_organmnist,
    download_medmnist,
    load_organmnist_data,
    load_medmnist_data,
    get_image_from_array,
    get_medmnist_label_names,
    get_human_readable_label,
    DEFAULT_DATA_DIR,
)


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def use_permanent_data():
    """Use the permanent data directory for tests that don't need cleanup."""
    return str(DEFAULT_DATA_DIR)


def test_download_organmnist_temp_dir(temp_data_dir):
    """Test downloading OrganSMNIST dataset to temporary directory."""
    download_organmnist(root=temp_data_dir)
    # Check that files were created
    data_path = Path(temp_data_dir) / "organsmnist.npz"
    assert data_path.exists()


def test_download_organmnist_permanent_dir():
    """Test downloading OrganSMNIST dataset to permanent directory."""
    # This will use the default permanent directory
    download_organmnist()
    # Check that files were created in permanent directory
    data_path = DEFAULT_DATA_DIR / "organsmnist.npz"
    assert data_path.exists()


def test_download_organmnist_already_exists():
    """Test that download_organmnist skips download if data already exists."""
    # First ensure data exists
    download_organmnist()
    
    # Check that file exists
    data_path = DEFAULT_DATA_DIR / "organsmnist.npz"
    assert data_path.exists()
    
    # Get modification time
    mtime = data_path.stat().st_mtime
    
    # Call download again - should skip
    download_organmnist()
    
    # Check that modification time hasn't changed (file wasn't re-downloaded)
    assert data_path.stat().st_mtime == mtime


def test_load_organmnist_train_data():
    """Test loading training data from permanent directory."""
    images, labels = load_organmnist_data(split="train")
    
    # Check shapes
    assert images.ndim == 3  # (N, 28, 28)
    assert images.shape[1:] == (28, 28)
    assert labels.ndim == 1
    assert len(images) == len(labels)
    
    # Check data types
    assert images.dtype == np.uint8 or images.dtype == np.float32
    assert labels.dtype in [np.int64, np.int32, np.uint8]


def test_load_organmnist_test_data():
    """Test loading test data from permanent directory."""
    images, labels = load_organmnist_data(split="test")
    
    # Check shapes
    assert images.ndim == 3  # (N, 28, 28)
    assert images.shape[1:] == (28, 28)
    assert labels.ndim == 1
    assert len(images) == len(labels)


def test_load_organmnist_with_temp_dir(temp_data_dir):
    """Test loading data from a specified temporary directory."""
    download_organmnist(root=temp_data_dir)
    images, labels = load_organmnist_data(split="train", root=temp_data_dir)
    
    # Check shapes
    assert images.ndim == 3  # (N, 28, 28)
    assert images.shape[1:] == (28, 28)
    assert labels.ndim == 1
    assert len(images) == len(labels)


def test_get_image_from_array():
    """Test converting numpy array to PIL Image."""
    # Test grayscale image
    grayscale_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
    pil_image = get_image_from_array(grayscale_array)
    
    assert isinstance(pil_image, Image.Image)
    assert pil_image.size == (28, 28)
    assert pil_image.mode == "L"  # Grayscale mode
    
    # Test RGB image
    rgb_array = np.random.randint(0, 255, size=(28, 28, 3), dtype=np.uint8)
    pil_image = get_image_from_array(rgb_array)
    
    assert isinstance(pil_image, Image.Image)
    assert pil_image.size == (28, 28)
    assert pil_image.mode == "RGB"


def test_download_medmnist_pathmnist(temp_data_dir):
    """Test downloading PathMNIST dataset."""
    download_medmnist(dataset_name="PathMNIST", root=temp_data_dir)
    data_path = Path(temp_data_dir) / "pathmnist.npz"
    assert data_path.exists()


def test_load_medmnist_data_pathmnist(temp_data_dir):
    """Test loading PathMNIST data."""
    download_medmnist(dataset_name="PathMNIST", root=temp_data_dir)
    images, labels = load_medmnist_data(dataset_name="PathMNIST", split="train", root=temp_data_dir)
    
    # Check shapes - PathMNIST is RGB
    assert images.ndim == 4  # (N, 28, 28, 3)
    assert images.shape[1:] == (28, 28, 3)
    assert labels.ndim == 1
    assert len(images) == len(labels)


def test_get_medmnist_label_names():
    """Test getting label names for OrganSMNIST."""
    # Ensure dataset exists
    download_medmnist(dataset_name="OrganSMNIST")
    
    label_names = get_medmnist_label_names(dataset_name="OrganSMNIST")
    assert isinstance(label_names, dict)
    assert len(label_names) == 11  # OrganSMNIST has 11 classes
    assert all(isinstance(k, int) for k in label_names.keys())
    assert all(isinstance(v, str) for v in label_names.values())


def test_get_human_readable_label():
    """Test getting human readable labels."""
    # Ensure dataset exists
    download_medmnist(dataset_name="OrganSMNIST")
    
    label = get_human_readable_label(0, dataset_name="OrganSMNIST")
    assert isinstance(label, str)
    assert len(label) > 0
    
    # Test unknown label
    unknown_label = get_human_readable_label(999, dataset_name="OrganSMNIST")
    assert "Unknown" in unknown_label
