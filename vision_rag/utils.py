"""Utility functions for vision RAG system."""

import base64
import binascii
import io
from PIL import Image, UnidentifiedImageError


def encode_image_to_base64(image: Image.Image) -> str:
    """
    Encode a PIL Image to a base64 string.
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 encoded image string
    """
    buffer = io.BytesIO()
    # Save image to buffer in PNG format to preserve quality
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_bytes = buffer.read()
    return base64.b64encode(image_bytes).decode('utf-8')


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
    except binascii.Error as e:
        raise ValueError(f"Invalid base64 string: {str(e)}") from e
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except UnidentifiedImageError as e:
        raise ValueError(f"Cannot identify image format: {str(e)}") from e
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}") from e
