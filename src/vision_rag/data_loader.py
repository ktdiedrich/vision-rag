"""Data loading module for OrganSMNIST dataset."""

from typing import Tuple
import numpy as np
from medmnist import OrganSMNIST
from PIL import Image


def download_organmnist(root: str = "./data") -> None:
    """
    Download the OrganSMNIST dataset.
    
    Args:
        root: Root directory to save the dataset
    """
    # Download training set
    OrganSMNIST(split="train", download=True, root=root)
    # Download test set
    OrganSMNIST(split="test", download=True, root=root)


def load_organmnist_data(
    split: str = "train", root: str = "./data"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load OrganSMNIST dataset.
    
    Args:
        split: Dataset split ('train' or 'test')
        root: Root directory where dataset is saved
        
    Returns:
        Tuple of (images, labels) where images is an array of shape (N, 28, 28, 3)
        and labels is an array of shape (N,)
    """
    dataset = OrganSMNIST(split=split, download=False, root=root)
    images = dataset.imgs  # Shape: (N, 28, 28, 3)
    labels = dataset.labels.squeeze()  # Shape: (N,)
    return images, labels


def get_image_from_array(image_array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    
    Args:
        image_array: Numpy array of shape (28, 28, 3)
        
    Returns:
        PIL Image
    """
    return Image.fromarray(image_array.astype(np.uint8))
