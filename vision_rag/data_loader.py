"""Data loading module for MedMNIST datasets."""

from pathlib import Path
from typing import Tuple
import numpy as np
import medmnist
from PIL import Image

from .config import MEDMNIST_DATASET, MEDMNIST_SIZE, get_dataset_config

# Default permanent data directory
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def download_medmnist(dataset_name: str = None, root: str = None, size: int = None) -> None:
    """
    Download a MedMNIST dataset if it doesn't already exist.
    
    Args:
        dataset_name: Name of the MedMNIST dataset (e.g., 'OrganSMNIST', 'PathMNIST').
                     If None, uses VISION_RAG_DATASET from environment.
        root: Root directory to save the dataset. If None, uses default permanent directory.
        size: Image size to download (28, 64, 128, or 224). If None, uses MEDMNIST_SIZE from config (default: 224).
    """
    if dataset_name is None:
        dataset_name = MEDMNIST_DATASET
    
    if root is None:
        root = DEFAULT_DATA_DIR
    
    if size is None:
        size = MEDMNIST_SIZE
    
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    
    # Get dataset configuration
    config = get_dataset_config(dataset_name)
    
    # Check if dataset file already exists for this size
    if size == 28:
        dataset_filename = dataset_name.lower() + ".npz"
    else:
        dataset_filename = f"{dataset_name.lower()}_{size}.npz"
    dataset_path = root_path / dataset_filename
    
    if dataset_path.exists():
        print(f"{dataset_name} dataset (size={size}) already exists in {root_path}")
        return
    
    print(f"Downloading {dataset_name} dataset (size={size}) to {root_path}")
    # Download both training and test data
    dataset_class = getattr(medmnist, config["class_name"])
    dataset_class(split="train", download=True, root=str(root_path), size=size)
    print("Download completed!")


def load_medmnist_data(
    dataset_name: str = None,
    split: str = "train",
    root: str = None,
    size: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a MedMNIST dataset.
    
    Args:
        dataset_name: Name of the MedMNIST dataset (e.g., 'OrganSMNIST', 'PathMNIST').
                     If None, uses VISION_RAG_DATASET from environment.
        split: Dataset split ('train', 'test', or 'val')
        root: Root directory where dataset is saved. If None, uses default permanent directory.
        size: Image size to load (28, 64, 128, or 224). If None, uses MEDMNIST_SIZE from config (default: 224).
        
    Returns:
        Tuple of (images, labels) where images is an array of shape (N, H, W) or (N, H, W, C)
        and labels is an array of shape (N,)
    """
    if dataset_name is None:
        dataset_name = MEDMNIST_DATASET
    
    if root is None:
        root = DEFAULT_DATA_DIR
    
    if size is None:
        size = MEDMNIST_SIZE
    
    # Ensure data is downloaded
    download_medmnist(dataset_name, root, size)
    
    # Get dataset class and load
    config = get_dataset_config(dataset_name)
    dataset_class = getattr(medmnist, config["class_name"])
    dataset = dataset_class(split=split, download=False, root=str(root), size=size)
    
    images = dataset.imgs  # Shape: (N, H, W) or (N, H, W, C)
    labels = dataset.labels.squeeze()  # Shape: (N,)
    return images, labels


def load_organmnist_data(
    split: str = "train", root: str = None, size: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load OrganSMNIST dataset.
    
    Args:
        split: Dataset split ('train' or 'test')
        root: Root directory where dataset is saved. If None, uses default permanent directory.
        size: Image size to load (28, 64, 128, or 224). If None, uses MEDMNIST_SIZE from config (default: 224).
        
    Returns:
        Tuple of (images, labels) where images is an array of shape (N, H, W)
        and labels is an array of shape (N,)
        
    Note:
        This function is kept for backward compatibility.
        Use load_medmnist_data() for more flexibility.
    """
    return load_medmnist_data(dataset_name="OrganSMNIST", split=split, root=root, size=size)


def get_image_from_array(image_array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    
    Args:
        image_array: Numpy array of shape (H, W) for grayscale or (H, W, 3) for RGB
        
    Returns:
        PIL Image
    """
    image_array = image_array.astype(np.uint8)
    if image_array.ndim == 2:
        # Grayscale image
        return Image.fromarray(image_array, mode='L')
    elif image_array.ndim == 3:
        # RGB image
        return Image.fromarray(image_array, mode='RGB')
    else:
        raise ValueError(f"Unsupported image array shape: {image_array.shape}")


def get_medmnist_label_names(dataset_name: str = None, root: str = None, size: int = None) -> dict:
    """
    Get human readable label names for a MedMNIST dataset.
    
    Args:
        dataset_name: Name of the MedMNIST dataset. If None, uses VISION_RAG_DATASET from environment.
        root: Root directory where dataset is saved. If None, uses default permanent directory.
        size: Image size (28, 64, 128, or 224). If None, uses MEDMNIST_SIZE from config (default: 224).
    
    Returns:
        Dictionary mapping label indices to human readable names
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist and needs to be downloaded
    """
    if dataset_name is None:
        dataset_name = MEDMNIST_DATASET
    
    if root is None:
        root = DEFAULT_DATA_DIR
    
    if size is None:
        size = MEDMNIST_SIZE
    
    # Check if dataset file exists
    if size == 28:
        dataset_filename = dataset_name.lower() + ".npz"
    else:
        dataset_filename = f"{dataset_name.lower()}_{size}.npz"
    dataset_path = Path(root) / dataset_filename
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} dataset (size={size}) not found at {dataset_path}. "
            f"Please download the dataset first by calling download_medmnist('{dataset_name}', size={size}) "
            f"or load_medmnist_data('{dataset_name}', size={size})."
        )
    
    # Get dataset class and load metadata
    config = get_dataset_config(dataset_name)
    dataset_class = getattr(medmnist, config["class_name"])
    dataset = dataset_class(split="train", download=False, root=str(root), size=size)
    
    if hasattr(dataset, 'info') and 'label' in dataset.info:
        # Convert string keys to integers
        return {int(k): v for k, v in dataset.info['label'].items()}
    else:
        # Return generic labels if no info available
        n_classes = config.get("n_classes", 10)
        return {i: f"Class {i}" for i in range(n_classes)}


def get_organmnist_label_names() -> dict:
    """
    Get human readable label names for OrganSMNIST dataset.
    
    Returns:
        Dictionary mapping label indices to human readable names
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist and needs to be downloaded
        
    Note:
        This function is kept for backward compatibility.
        Use get_medmnist_label_names() for more flexibility.
    """
    return get_medmnist_label_names(dataset_name="OrganSMNIST")


def get_human_readable_label(label_index: int, dataset_name: str = None, root: str = None, size: int = None) -> str:
    """
    Get human readable label for a given label index.
    
    Args:
        label_index: Numeric label index
        dataset_name: Name of the MedMNIST dataset. If None, uses VISION_RAG_DATASET from environment.
        root: Root directory where dataset is saved. If None, uses default permanent directory.
        size: Image size (28, 64, 128, or 224). If None, uses MEDMNIST_SIZE from config (default: 224).
        
    Returns:
        Human readable label name
    """
    label_names = get_medmnist_label_names(dataset_name, root, size)
    return label_names.get(label_index, f"Unknown ({label_index})")

