"""Image search functionality using RAG store."""

from typing import List, Union, Dict
import json
from collections import Counter, defaultdict
import numpy as np
from PIL import Image

from .encoder import ImageEncoderProtocol
from .rag_store import ChromaRAGStore


class ImageSearcher:
    """Search for similar images using an image encoder and ChromaDB.

    Also provides a simple k-nearest-neighbor classifier that assigns a
    predicted label based on the majority label among the N nearest
    neighbors returned by the RAG store.
    """
    
    def __init__(
        self,
        encoder: ImageEncoderProtocol,
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

    def classify(
        self,
        query_image: Union[Image.Image, np.ndarray],
        n_results: int = 5,
    ) -> Dict[str, Union[int, float, dict]]:
        """
        Classify a query image using k-nearest neighbors (majority vote).

        Args:
            query_image: Query image as PIL Image or numpy array
            n_results: Number of nearest neighbors to use for classification

        Returns:
            A dictionary containing:
            - label: Predicted label (most common label among neighbors)
            - count: Number of neighbors with the predicted label
            - confidence: fraction of neighbors with predicted label (count / n_results)
            - neighbors: raw neighbor search results (ids, distances, metadatas)
        """
        # Use the existing search method to retrieve neighbors
        results = self.search(query_image, n_results=n_results)

        # Extract labels from metadata
        labels = []
        for meta in results.get("metadatas", []):
            label_value = meta.get("label", None)
            if label_value is None:
                # fall back to index if label not found
                label_value = meta.get("index", None)

            # Normalize label: handle JSON strings, lists, numpy arrays
            if isinstance(label_value, str):
                try:
                    # Try JSON decode in case of stringified list
                    parsed = json.loads(label_value)
                    if isinstance(parsed, (list, tuple)) and parsed:
                        label_value = int(parsed[0])
                    else:
                        label_value = int(parsed)
                except Exception:
                    try:
                        label_value = int(label_value)
                    except Exception:
                        # keep as-is (string)
                        pass
            elif isinstance(label_value, (list, tuple)):
                label_value = int(label_value[0])
            elif isinstance(label_value, (np.ndarray,)):
                if label_value.size > 0:
                    label_value = int(label_value.flat[0])

            labels.append(label_value)

        if not labels:
            # No label info available
            return {
                "label": None,
                "count": 0,
                "confidence": 0.0,
                "neighbors": results,
            }

        # Count labels
        counts = Counter(labels)
        most_common = counts.most_common()
        top_count = most_common[0][1]
        # Get all labels that tie for top count
        top_labels = [label for label, cnt in most_common if cnt == top_count]

        if len(top_labels) == 1:
            predicted_label = top_labels[0]
        else:
            # Tie-breaker: pick the tied label with the smallest average distance
            avg_dist = {}
            for tlabel in top_labels:
                dist_sum = 0.0
                dist_count = 0
                for lbl, d in zip(labels, results.get("distances", [])):
                    if lbl == tlabel and d is not None:
                        dist_sum += float(d)
                        dist_count += 1
                avg_dist[tlabel] = dist_sum / dist_count if dist_count > 0 else float("inf")
            predicted_label = min(avg_dist.keys(), key=lambda k: avg_dist[k])

        return {
            "label": predicted_label,
            "count": int(counts[predicted_label]),
            "confidence": counts[predicted_label] / float(len(labels)),
            "neighbors": results,
        }

    def batch_classify(
        self,
        query_images: List[Union[Image.Image, np.ndarray]],
        n_results: int = 5,
    ) -> List[Dict[str, Union[int, float, dict]]]:
        """
        Classify a list of query images using k-nearest neighbors.
        Returns a list of classification results (same structure as `classify`).
        """
        return [self.classify(q, n_results=n_results) for q in query_images]
