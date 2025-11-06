"""Utility functions for vision RAG system."""

import base64
import io
from PIL import Image


def decode_base64_image(image_base64: str) -> Image.Image:
    """
    Decode a base64 encoded image string to a PIL Image.
    
    Args:
        image_base64: Base64 encoded image string
        
    Returns:
        PIL Image object
        
    Raises:
        ValueError: If the base64 string is invalid or cannot be decoded as an image
    """
    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}") from e
