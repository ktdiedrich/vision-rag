"""Data loading module for OrganSMNIST dataset."""

from pathlib import Path
from typing import Tuple
import numpy as np
from medmnist import OrganSMNIST
from PIL import Image

# Default permanent data directory
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def download_organmnist(root: str = None) -> None:
    """
    Download the OrganSMNIST dataset if it doesn't already exist.
    
    Args:
        root: Root directory to save the dataset. If None, uses default permanent directory.
    """
    if root is None:
        root = DEFAULT_DATA_DIR
    
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset file already exists
    organmnist_data_path = root_path / "organsmnist.npz"
    
    if organmnist_data_path.exists():
        print(f"OrganSMNIST dataset already exists in {root_path}")
        return
    
    print(f"Downloading OrganSMNIST dataset to {root_path}")
    # Download both training and test data (creates organsmnist.npz)
    OrganSMNIST(split="train", download=True, root=str(root_path))
    print("Download completed!")


def load_organmnist_data(
    split: str = "train", root: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load OrganSMNIST dataset.
    
    Args:
        split: Dataset split ('train' or 'test')
        root: Root directory where dataset is saved. If None, uses default permanent directory.
        
    Returns:
        Tuple of (images, labels) where images is an array of shape (N, 28, 28)
        and labels is an array of shape (N,)
    """
    if root is None:
        root = DEFAULT_DATA_DIR
    
    # Ensure data is downloaded
    download_organmnist(root)
    
    dataset = OrganSMNIST(split=split, download=False, root=str(root))
    images = dataset.imgs  # Shape: (N, 28, 28)
    labels = dataset.labels.squeeze()  # Shape: (N,)
    return images, labels


def get_image_from_array(image_array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    
    Args:
        image_array: Numpy array of shape (28, 28) for grayscale or (28, 28, 3) for RGB
        
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


def get_organmnist_label_names() -> dict:
    """
    Get human readable label names for OrganSMNIST dataset.
    
    Returns:
        Dictionary mapping label indices to human readable names
    """
    dataset = OrganSMNIST(split="train", download=False, root=str(DEFAULT_DATA_DIR))
    if hasattr(dataset, 'info') and 'label' in dataset.info:
        # Convert string keys to integers
        return {int(k): v for k, v in dataset.info['label'].items()}
    else:
        # Fallback mapping if dataset info is not available
        return {
            0: 'bladder', 1: 'femur-left', 2: 'femur-right', 3: 'heart', 
            4: 'kidney-left', 5: 'kidney-right', 6: 'liver', 7: 'lung-left', 
            8: 'lung-right', 9: 'pancreas', 10: 'spleen'
        }


def get_human_readable_label(label_index: int) -> str:
    """
    Get human readable label for a given label index.
    
    Args:
        label_index: Numeric label index
        
    Returns:
        Human readable label name
    """
    label_names = get_organmnist_label_names()
    return label_names.get(label_index, f"Unknown ({label_index})")
