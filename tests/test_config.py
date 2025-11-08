"""Tests for configuration module."""

import pytest
from vision_rag.config import (
    CLIP_MODEL_NAME,
    MEDMNIST_DATASET,
    get_dataset_config,
    list_available_datasets,
    AVAILABLE_DATASETS,
)


def test_default_config_values():
    """Test default configuration values."""
    assert CLIP_MODEL_NAME == "clip-ViT-B-32"
    assert MEDMNIST_DATASET == "OrganSMNIST"


def test_list_available_datasets():
    """Test listing available datasets."""
    datasets = list_available_datasets()
    assert isinstance(datasets, list)
    assert len(datasets) > 0
    assert "OrganSMNIST" in datasets
    assert "PathMNIST" in datasets
    assert "ChestMNIST" in datasets


def test_get_dataset_config_default():
    """Test getting dataset config with default."""
    config = get_dataset_config()
    assert config["class_name"] == "OrganSMNIST"
    assert config["n_classes"] == 11
    assert config["image_size"] == 224


def test_get_dataset_config_specific():
    """Test getting dataset config for specific dataset."""
    config = get_dataset_config("PathMNIST")
    assert config["class_name"] == "PathMNIST"
    assert config["n_classes"] == 9
    assert config["image_size"] == 224
    assert config["channels"] == 3


def test_get_dataset_config_invalid():
    """Test getting dataset config for invalid dataset."""
    with pytest.raises(ValueError, match="not supported"):
        get_dataset_config("InvalidDataset")


def test_available_datasets_structure():
    """Test structure of available datasets."""
    for name, config in AVAILABLE_DATASETS.items():
        assert "class_name" in config
        assert "description" in config
        assert "n_classes" in config
        assert "image_size" in config
        assert "channels" in config
        
        # Verify types
        assert isinstance(config["class_name"], str)
        assert isinstance(config["description"], str)
        assert isinstance(config["n_classes"], int)
        assert isinstance(config["image_size"], int)
        assert isinstance(config["channels"], int)
        
        # Verify reasonable values
        assert config["n_classes"] > 0
        assert config["image_size"] == 224  # Default size for CLIP compatibility
        assert config["channels"] in [1, 3]  # Grayscale or RGB
