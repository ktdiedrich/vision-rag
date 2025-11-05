"""Image search functionality using RAG store."""

from typing import List, Union, Dict
import numpy as np
from PIL import Image

from .encoder import CLIPImageEncoder
from .rag_store import ChromaRAGStore


class ImageSearcher:
    """Search for similar images using CLIP embeddings and ChromaDB."""
    
    def __init__(
        self,
        encoder: CLIPImageEncoder,
        rag_store: ChromaRAGStore,
    ):
        """
        Initialize the image searcher.
        
        Args:
            encoder: CLIP image encoder
            rag_store: ChromaDB RAG store
        """
        self.encoder = encoder
        self.rag_store = rag_store
    
    def search(
        self,
        query_image: Union[Image.Image, np.ndarray],
        n_results: int = 5,
    ) -> Dict:
        """
        Search for similar images in the RAG store.
        
        Args:
            query_image: Query image as PIL Image or numpy array
            n_results: Number of similar images to retrieve
            
        Returns:
            Dictionary with search results containing:
            - ids: List of matching image IDs
            - distances: List of distances to query
            - metadatas: List of metadata for matched images
        """
        # Encode the query image
        query_embedding = self.encoder.encode_image(query_image)
        
        # Search in the RAG store
        results = self.rag_store.search(
            query_embedding=query_embedding,
            n_results=n_results,
        )
        
        return results
    
    def batch_search(
        self,
        query_images: List[Union[Image.Image, np.ndarray]],
        n_results: int = 5,
    ) -> List[Dict]:
        """
        Search for similar images for multiple queries.
        
        Args:
            query_images: List of query images
            n_results: Number of similar images to retrieve per query
            
        Returns:
            List of dictionaries, one for each query image
        """
        results = []
        for query_image in query_images:
            result = self.search(query_image, n_results=n_results)
            results.append(result)
        
        return results
