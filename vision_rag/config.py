"""Configuration module for Vision RAG system."""

import os
from typing import Dict, Any


# Environment variable configuration
CLIP_MODEL_NAME = os.getenv("VISION_RAG_CLIP_MODEL", "clip-ViT-B-32")
MEDMNIST_DATASET = os.getenv("VISION_RAG_DATASET", "OrganSMNIST")
COLLECTION_NAME = os.getenv("VISION_RAG_COLLECTION_NAME", "vision_rag")
PERSIST_DIRECTORY = os.getenv("VISION_RAG_PERSIST_DIR", "./chroma_db_api")
# Image storage size - images will be resized to this size before storage
# Default is 224 (CLIP's input size) for optimal quality
# Set to None or empty string to disable resizing
_image_size_str = os.getenv("VISION_RAG_IMAGE_SIZE", "224")
IMAGE_SIZE = int(_image_size_str) if _image_size_str and _image_size_str.lower() != "none" else None
# MedMNIST dataset download size - images are available in 28, 64, 128, or 224
# Default is 224 to match CLIP's input size for best quality
_medmnist_size_str = os.getenv("VISION_RAG_MEDMNIST_SIZE", "224")
MEDMNIST_SIZE = int(_medmnist_size_str) if _medmnist_size_str else 224


# Available MedMNIST datasets with their characteristics
# Note: image_size reflects the default MEDMNIST_SIZE (224)
# Datasets are available in 28, 64, 128, and 224 pixel sizes
AVAILABLE_DATASETS: Dict[str, Dict[str, Any]] = {
    # 2D datasets (default 224x224, also available in 28, 64, 128)
    "PathMNIST": {
        "class_name": "PathMNIST",
        "description": "Colon pathology images",
        "n_classes": 9,
        "image_size": 224,
        "channels": 3,
    },
    "ChestMNIST": {
        "class_name": "ChestMNIST",
        "description": "Chest X-ray images",
        "n_classes": 14,
        "image_size": 224,
        "channels": 1,
    },
    "DermaMNIST": {
        "class_name": "DermaMNIST",
        "description": "Dermatoscope images of skin lesions",
        "n_classes": 7,
        "image_size": 224,
        "channels": 3,
    },
    "OCTMNIST": {
        "class_name": "OCTMNIST",
        "description": "Retinal OCT images",
        "n_classes": 4,
        "image_size": 224,
        "channels": 1,
    },
    "PneumoniaMNIST": {
        "class_name": "PneumoniaMNIST",
        "description": "Chest X-ray images for pneumonia detection",
        "n_classes": 2,
        "image_size": 224,
        "channels": 1,
    },
    "RetinaMNIST": {
        "class_name": "RetinaMNIST",
        "description": "Fundus camera images for diabetic retinopathy",
        "n_classes": 5,
        "image_size": 224,
        "channels": 3,
    },
    "BreastMNIST": {
        "class_name": "BreastMNIST",
        "description": "Breast ultrasound images",
        "n_classes": 2,
        "image_size": 224,
        "channels": 1,
    },
    "BloodMNIST": {
        "class_name": "BloodMNIST",
        "description": "Blood cell microscope images",
        "n_classes": 8,
        "image_size": 224,
        "channels": 3,
    },
    "TissueMNIST": {
        "class_name": "TissueMNIST",
        "description": "Kidney cortex cells",
        "n_classes": 8,
        "image_size": 224,
        "channels": 1,
    },
    "OrganAMNIST": {
        "class_name": "OrganAMNIST",
        "description": "Organ images (axial view)",
        "n_classes": 11,
        "image_size": 224,
        "channels": 1,
    },
    "OrganCMNIST": {
        "class_name": "OrganCMNIST",
        "description": "Organ images (coronal view)",
        "n_classes": 11,
        "image_size": 224,
        "channels": 1,
    },
    "OrganSMNIST": {
        "class_name": "OrganSMNIST",
        "description": "Organ images (sagittal view)",
        "n_classes": 11,
        "image_size": 224,
        "channels": 1,
    },
}


def get_dataset_config(dataset_name: str = None) -> Dict[str, Any]:
    """
    Get configuration for a specific MedMNIST dataset.
    
    Args:
        dataset_name: Name of the dataset. If None, uses MEDMNIST_DATASET from environment.
        
    Returns:
        Dictionary with dataset configuration
        
    Raises:
        ValueError: If dataset name is not supported
    """
    if dataset_name is None:
        dataset_name = MEDMNIST_DATASET
    
    if dataset_name not in AVAILABLE_DATASETS:
        available = ", ".join(AVAILABLE_DATASETS.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. "
            f"Available datasets: {available}"
        )
    
    return AVAILABLE_DATASETS[dataset_name]


def list_available_datasets() -> list:
    """
    List all available MedMNIST datasets.
    
    Returns:
        List of dataset names
    """
    return list(AVAILABLE_DATASETS.keys())
