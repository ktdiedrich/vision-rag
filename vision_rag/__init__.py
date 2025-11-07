"""Vision RAG - A retrieval-augmented generation system for medical images."""

from .config import (
    CLIP_MODEL_NAME,
    MEDMNIST_DATASET,
    COLLECTION_NAME,
    PERSIST_DIRECTORY,
    get_dataset_config,
    list_available_datasets,
)
from .data_loader import (
    download_organmnist,
    download_medmnist,
    load_organmnist_data,
    load_medmnist_data,
    get_image_from_array,
    get_organmnist_label_names,
    get_medmnist_label_names,
    get_human_readable_label,
)
from .encoder import CLIPImageEncoder
from .rag_store import ChromaRAGStore
from .search import ImageSearcher
from .utils import decode_base64_image, encode_image_to_base64
from .visualization import RAGVisualizer


__all__ = [
    # Configuration
    "CLIP_MODEL_NAME",
    "MEDMNIST_DATASET",
    "COLLECTION_NAME",
    "PERSIST_DIRECTORY",
    "get_dataset_config",
    "list_available_datasets",
    # Data loading
    "download_organmnist",
    "download_medmnist",
    "load_organmnist_data",
    "load_medmnist_data",
    "get_image_from_array",
    "get_organmnist_label_names",
    "get_medmnist_label_names",
    "get_human_readable_label",
    # Core components
    "CLIPImageEncoder",
    "ChromaRAGStore",
    "ImageSearcher",
    "decode_base64_image",
    "encode_image_to_base64",
    "RAGVisualizer",
]

