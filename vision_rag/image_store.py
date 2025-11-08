"""Image file storage manager for Vision RAG."""

from pathlib import Path
from typing import Optional, Union
import hashlib
from PIL import Image
import numpy as np


class ImageFileStore:
    """
    Manager for storing and retrieving images from the file system.
    
    Images are saved to disk and their paths are stored in metadata.
    This allows the RAG system to work with file references rather than
    storing full images in the database.
    """
    
    def __init__(self, storage_dir: str = "./image_store", image_size: Optional[int] = None):
        """
        Initialize the image file store.
        
        Args:
            storage_dir: Directory where images will be saved
            image_size: Optional target size for images (width and height). 
                       If provided, images will be resized to (image_size, image_size) before saving.
                       If None, images are saved at their original size.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
    
    def _generate_image_id(self, image: Union[Image.Image, np.ndarray]) -> str:
        """
        Generate a unique ID for an image based on its content.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Unique hash string for the image
        """
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Generate hash from image data
        img_bytes = img_array.tobytes()
        hash_obj = hashlib.sha256(img_bytes)
        return hash_obj.hexdigest()[:16]  # Use first 16 chars of hash
    
    def save_image(
        self,
        image: Union[Image.Image, np.ndarray],
        image_id: Optional[str] = None,
        prefix: str = "img"
    ) -> str:
        """
        Save an image to the file store.
        
        Args:
            image: PIL Image or numpy array to save
            image_id: Optional custom ID for the image
            prefix: Prefix for the filename (default: "img")
            
        Returns:
            Full path (absolute or relative, depending on storage_dir) to the saved image as a string
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            # Handle grayscale images
            if image.ndim == 2:
                pil_image = Image.fromarray(image.astype(np.uint8), mode='L')
            # Handle RGB images
            elif image.ndim == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(image.astype(np.uint8), mode='RGB')
            # Handle RGBA images
            elif image.ndim == 3 and image.shape[2] == 4:
                pil_image = Image.fromarray(image.astype(np.uint8), mode='RGBA')
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        else:
            pil_image = image
        
        # Generate image ID if not provided (before resizing to ensure consistency)
        if image_id is None:
            image_id = self._generate_image_id(pil_image)
        
        # Resize image if target size is configured
        if self.image_size is not None:
            # Use LANCZOS for high-quality downsampling
            pil_image = pil_image.resize(
                (self.image_size, self.image_size),
                Image.Resampling.LANCZOS
            )
        
        # Create filename with appropriate extension
        filename = f"{prefix}_{image_id}.png"
        filepath = self.storage_dir / filename
        
        # If file exists, reuse the path; otherwise, save the image
        if not filepath.exists():
            pil_image.save(filepath)
        
        # Return relative path from storage directory
        return str(filepath)
    
    def load_image(self, image_path: str) -> Image.Image:
        """
        Load an image from the file store.
        
        Args:
            image_path: Path to the image file (relative or absolute)
            
        Returns:
            PIL Image object
        """
        path = Path(image_path)
        
        # Handle both relative and absolute paths
        if not path.is_absolute():
            path = Path.cwd() / path
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        return Image.open(path)
    
    def delete_image(self, image_path: str) -> bool:
        """
        Delete an image from the file store.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if deleted successfully, False otherwise
        """
        path = Path(image_path)
        
        if not path.is_absolute():
            path = Path.cwd() / path
        
        if path.exists():
            path.unlink()
            return True
        return False
    
    def image_exists(self, image_path: str) -> bool:
        """
        Check if an image exists in the file store.
        
        Args:
            image_path: Path to check
            
        Returns:
            True if image exists, False otherwise
        """
        path = Path(image_path)
        
        if not path.is_absolute():
            path = Path.cwd() / path
        
        return path.exists()
    
    def clear(self) -> int:
        """
        Remove all images from the store.
        
        Returns:
            Number of images deleted
        """
        count = 0
        for filepath in self.storage_dir.glob("*.png"):
            filepath.unlink()
            count += 1
        for filepath in self.storage_dir.glob("*.jpg"):
            filepath.unlink()
            count += 1
        return count
    
    def count(self) -> int:
        """
        Count the number of images in the store.
        
        Returns:
            Number of image files
        """
        return len(list(self.storage_dir.glob("*.png"))) + \
               len(list(self.storage_dir.glob("*.jpg")))
