"""Image encoder using CLIP model."""

from typing import List, Union
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

# Default image size for dummy images (OrganSMNIST dataset dimensions)
DEFAULT_IMAGE_SIZE = 28


class CLIPImageEncoder:
    """Image encoder using CLIP ViT-B-32 model."""
    
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        """
        Initialize the CLIP encoder.
        
        Args:
            model_name: Name of the CLIP model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
    def encode_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Encode a single image to an embedding vector.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Embedding vector as numpy array
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        # Encode the image
        embedding = self.model.encode(image)
        return embedding
    
    def encode_images(self, images: List[Union[Image.Image, np.ndarray]]) -> np.ndarray:
        """
        Encode multiple images to embedding vectors.
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            Array of embedding vectors with shape (N, embedding_dim)
        """
        # Convert numpy arrays to PIL Images if needed
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img.astype(np.uint8)))
            else:
                pil_images.append(img)
        
        # Encode all images
        embeddings = self.model.encode(pil_images)
        return embeddings
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        # Try to get dimension from model first
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is not None:
            return dimension
        
        # If not available, encode a dummy image to get the dimension
        dummy_image = Image.fromarray(
            np.zeros((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), dtype=np.uint8), mode='L'
        )
        embedding = self.model.encode(dummy_image)
        return embedding.shape[0]
