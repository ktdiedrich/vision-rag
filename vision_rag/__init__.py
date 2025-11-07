"""Vision RAG - A retrieval-augmented generation system for medical images."""

import os
from .data_loader import download_organmnist, load_organmnist_data, get_image_from_array, get_organmnist_label_names, get_human_readable_label
from .encoder import CLIPImageEncoder
from .rag_store import ChromaRAGStore
from .search import ImageSearcher
from .utils import decode_base64_image
from .visualization import RAGVisualizer


__all__ = [
    "download_organmnist",
    "load_organmnist_data",
    "get_image_from_array",
    "get_organmnist_label_names",
    "get_human_readable_label",
    "CLIPImageEncoder",
    "ChromaRAGStore",
    "ImageSearcher",
    "decode_base64_image",
    "RAGVisualizer",
]
# Configuration
CLIP_MODEL_NAME = os.getenv("VISION_RAG_CLIP_MODEL", "clip-ViT-B-32")
