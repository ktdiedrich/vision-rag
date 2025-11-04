"""Tests for data loading module."""

import pytest
import numpy as np
from PIL import Image
import tempfile
import shutil

from vision_rag.data_loader import (
    download_organmnist,
    load_organmnist_data,
    get_image_from_array,
)


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_download_organmnist(temp_data_dir):
    """Test downloading OrganSMNIST dataset."""
    download_organmnist(root=temp_data_dir)
    # If download succeeds without error, test passes


def test_load_organmnist_train_data(temp_data_dir):
    """Test loading training data."""
    download_organmnist(root=temp_data_dir)
    images, labels = load_organmnist_data(split="train", root=temp_data_dir)
    
    # Check shapes
    assert images.ndim == 4  # (N, 28, 28, 3)
    assert images.shape[1:] == (28, 28, 3)
    assert labels.ndim == 1
    assert len(images) == len(labels)
    
    # Check data types
    assert images.dtype == np.uint8 or images.dtype == np.float32
    assert labels.dtype in [np.int64, np.int32]


def test_load_organmnist_test_data(temp_data_dir):
    """Test loading test data."""
    download_organmnist(root=temp_data_dir)
    images, labels = load_organmnist_data(split="test", root=temp_data_dir)
    
    # Check shapes
    assert images.ndim == 4  # (N, 28, 28, 3)
    assert images.shape[1:] == (28, 28, 3)
    assert labels.ndim == 1
    assert len(images) == len(labels)


def test_get_image_from_array():
    """Test converting numpy array to PIL Image."""
    # Create a test image array
    image_array = np.random.randint(0, 255, size=(28, 28, 3), dtype=np.uint8)
    
    # Convert to PIL Image
    pil_image = get_image_from_array(image_array)
    
    # Check type and size
    assert isinstance(pil_image, Image.Image)
    assert pil_image.size == (28, 28)
    assert pil_image.mode == "RGB"
