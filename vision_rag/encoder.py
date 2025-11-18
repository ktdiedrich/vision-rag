"""Image encoder using CLIP model."""

from typing import List, Union
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel
import torch

from .config import CLIP_MODEL_NAME

# Default image size for dummy images (MedMNIST dataset dimensions)
DEFAULT_IMAGE_SIZE = 224


class CLIPImageEncoder:
    """Image encoder using model from parameter."""
    
    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        """
        Initialize the encoder.
        
        Args:
            model_name: Name of the model to use
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


class DINOImageEncoder:
    """Image encoder using a DINO Vision Transformer model from Hugging Face.

    This uses `AutoImageProcessor`/`AutoModel` (Transformers) to create
    an image embedding by mean-pooling the ViT last_hidden_state across the
    sequence dimension. The returned embedding shape is (embedding_dim,).

    Example model name: ``facebook/dino-vits8``
    """

    def __init__(self, model_name: str = "facebook/dino-vits8", device: Union[str, torch.device] = None):
        self.model_name = model_name
        # Use AutoImageProcessor (newer API; replaces AutoFeatureExtractor)
        self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

    def _to_pil_list(self, images: List[Union[Image.Image, np.ndarray]]):
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                # Convert grayscale to RGB
                if img.ndim == 2:
                    img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
                elif img.shape[-1] == 1:
                    img = Image.fromarray(img.squeeze().astype(np.uint8)).convert("RGB")
                else:
                    img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
            else:
                if img.mode != "RGB":
                    img = img.convert("RGB")
            pil_images.append(img)
        return pil_images

    def encode_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        emb = self.encode_images([image])
        return emb[0]

    def encode_images(self, images: List[Union[Image.Image, np.ndarray]]) -> np.ndarray:
        pil_images = self._to_pil_list(images)

        # Extract pixel values and move tensors to device
        inputs = self.feature_extractor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden = outputs.last_hidden_state  # (batch, seq, hidden)
        # Mean pooling across seq dimension to produce fixed-size vector
        emb = last_hidden.mean(dim=1)

        return emb.cpu().numpy()

    @property
    def embedding_dimension(self) -> int:
        # Try to read hidden size from model config
        if hasattr(self.model.config, "hidden_size"):
            return int(self.model.config.hidden_size)

        # Fallback to compute with dummy image
        embedding = self.encode_image(np.zeros((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), dtype=np.uint8))
        return embedding.shape[0]
