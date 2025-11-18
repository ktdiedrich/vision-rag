"""Image encoder using CLIP model."""

from typing import List, Union, Protocol, runtime_checkable, Sequence
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel
import torch
import typing

from .config import CLIP_MODEL_NAME
from .config import ENCODER_TYPE, CLIP_MODEL_NAME, DINO_MODEL_NAME


# Default image size for dummy images (MedMNIST dataset dimensions)
DEFAULT_IMAGE_SIZE = 224


class BaseImageEncoder(ABC):
    """Base class for image encoders; provides shared helpers.

    Subclasses must implement `encode_images` and `embedding_dimension`.
    """

    def __init__(self, device: Union[str, torch.device] | None = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def _to_pil_list(self, images: Sequence[Union[Image.Image, np.ndarray]]) -> list[Image.Image]:
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                # Convert grayscale to RGB by default
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

    @abstractmethod
    def encode_images(self, images: List[Union[Image.Image, np.ndarray]]) -> np.ndarray:  # pragma: no cover - implemented in subclasses
        raise NotImplementedError()

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:  # pragma: no cover - implemented in subclasses
        raise NotImplementedError()


class CLIPImageEncoder(BaseImageEncoder):
    """Image encoder using model from parameter."""
    
    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        """
        Initialize the encoder.
        
        Args:
            model_name: Name of the model to use
        """
        super().__init__()
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

        # Convert PIL to numpy array (sentence-transformers expects arrays for image encoders)
        if isinstance(image, Image.Image):
            image_arr = np.asarray(image)
        else:
            image_arr = image

        # Encode the image. Provide a 1-element ndarray to match the encode() overload
        embedding = self.model.encode(np.expand_dims(image_arr, 0))
        # model.encode returns a list/array of embeddings; return first
        return embedding[0]
        return embedding
    
    def encode_images(self, images: Sequence[Union[Image.Image, np.ndarray]]) -> np.ndarray:
        """
        Encode multiple images to embedding vectors.
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            Array of embedding vectors with shape (N, embedding_dim)
        """
        # Convert numpy arrays to PIL Images if needed
        pil_images = self._to_pil_list(images)

        # Convert pil images to numpy arrays and stack into single ndarray
        image_arrays = np.asarray([np.asarray(img) for img in pil_images])

        # Encode all images (provide an ndarray to match the overloads)
        embeddings = self.model.encode(image_arrays)
        return embeddings
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        # Try to get dimension from model first
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is not None:
            return dimension
        
        # If not available, encode a dummy image to get the dimension
        dummy_image = np.zeros((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), dtype=np.uint8)
        embedding = self.model.encode(np.expand_dims(dummy_image, 0))
        return embedding.shape[0]


@runtime_checkable
class ImageEncoderProtocol(Protocol):
    """Protocol for image encoders.

    Implementations must provide encode_image, encode_images and the
    embedding_dimension property.
    """

    def encode_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray: ...

    def encode_images(self, images: Sequence[Union[Image.Image, np.ndarray]]) -> np.ndarray: ...

    @property
    def embedding_dimension(self) -> int: ...


class DINOImageEncoder(BaseImageEncoder):
    """Image encoder using a DINO Vision Transformer model from Hugging Face.

    This uses `AutoImageProcessor`/`AutoModel` (Transformers) to create
    an image embedding by mean-pooling the ViT last_hidden_state across the
    sequence dimension. The returned embedding shape is (embedding_dim,).

    Example model name: ``facebook/dino-vits8``
    """

    def __init__(self, model_name: str = "facebook/dino-vits8", device: Union[str, torch.device] | None = None):
        self.model_name = model_name
        # Use AutoImageProcessor (newer API; replaces AutoFeatureExtractor)
        self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        super().__init__(device=device)

        self.model.to(self.device)

    def _to_pil_list(self, images: Sequence[Union[Image.Image, np.ndarray]]) -> list[Image.Image]:
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

    def encode_images(self, images: Sequence[Union[Image.Image, np.ndarray]]) -> np.ndarray:
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


def build_encoder(encoder_type: str | None = None, **kwargs) -> ImageEncoderProtocol:
    """Factory: build an encoder by type.

    encoder_type: 'clip' (default) or 'dino'. Additional keyword args are passed
    through to the encoder constructors.
    """
    

    if encoder_type is None:
        encoder_type = ENCODER_TYPE

    encoder_type = encoder_type.lower()

    if encoder_type in ("clip", "sentence-transformers", "sentence_transformer"):
        # Use the SentenceTransformer-based CLIP encoder
        return CLIPImageEncoder(model_name=kwargs.pop("model_name", CLIP_MODEL_NAME), **kwargs)

    if encoder_type in ("dino", "dino-vits", "transformer", "transformers"):
        return DINOImageEncoder(model_name=kwargs.pop("model_name", DINO_MODEL_NAME), **kwargs)

    raise ValueError(f"Unknown encoder type: {encoder_type}")
