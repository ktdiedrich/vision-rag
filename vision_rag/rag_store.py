"""RAG store using ChromaDB."""

from typing import List, Optional
import numpy as np
import chromadb
from chromadb.config import Settings


class ChromaRAGStore:
    """RAG store using ChromaDB for storing and retrieving image embeddings."""
    
    def __init__(
        self,
        collection_name: str = "organmnist_images",
        persist_directory: str = "./chroma_db",
    ):
        """
        Initialize the ChromaDB RAG store.
        
        Args:
            collection_name: Name of the collection to store embeddings
            persist_directory: Directory to persist the ChromaDB data
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "OrganSMNIST image embeddings"}
        )
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        """
        Add embeddings to the RAG store.
        
        Args:
            embeddings: Array of embeddings with shape (N, embedding_dim)
            ids: Optional list of IDs for the embeddings
            metadatas: Optional list of metadata dictionaries
        """
        n_embeddings = len(embeddings)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"img_{i}" for i in range(n_embeddings)]
        
        # Ensure metadata is provided and non-empty
        if metadatas is None:
            metadatas = [{"index": i} for i in range(n_embeddings)]
        else:
            # Ensure each metadata dict is non-empty
            metadatas = [m if m else {"index": i} for i, m in enumerate(metadatas)]
        
        # Add embeddings to collection
        self.collection.add(
            embeddings=embeddings.tolist(),
            ids=ids,
            metadatas=metadatas,
        )
    
    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
    ) -> dict:
        """
        Search for similar embeddings in the RAG store.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            
        Returns:
            Dictionary containing search results with keys:
            - ids: List of matching IDs
            - distances: List of distances
            - metadatas: List of metadata dictionaries
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
        )
        
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        }
    
    def search_by_label(
        self,
        label: int,
        n_results: Optional[int] = None,
    ) -> dict:
        """
        Retrieve images with a specific label, optionally limited by n_results.
        
        Args:
            label: The label to search for
            n_results: Optional limit on number of results (default: all matching)
            
        Returns:
            Dictionary containing search results with keys:
            - ids: List of matching IDs
            - metadatas: List of metadata dictionaries
            - embeddings: List of embeddings (if available)
        """
        # Get all items with matching label
        results = self.collection.get(
            where={"label": label},
            limit=n_results,
            include=["metadatas", "embeddings"]
        )
        
        return {
            "ids": results["ids"] if results["ids"] else [],
            "metadatas": results["metadatas"] if results["metadatas"] else [],
            "embeddings": results.get("embeddings", []),
        }
    
    def count(self) -> int:
        """Get the number of embeddings in the store."""
        return self.collection.count()
    
    def clear(self) -> None:
        """Clear all embeddings from the store."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "OrganSMNIST image embeddings"}
        )
