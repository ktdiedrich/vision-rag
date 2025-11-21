"""Image search functionality using RAG store."""

from typing import List, Union, Dict, Any, Iterable, Tuple, cast, Optional
import json
from collections import Counter, defaultdict
import numpy as np
from PIL import Image

from .encoder import ImageEncoderProtocol
from .rag_store import ChromaRAGStore
from .visualization import RAGVisualizer


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
    ) -> Dict[str, Any]:
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

    def compute_confusion_and_metrics(
        self,
        true_labels: Iterable[Union[int, str]],
        predicted_labels: Iterable[Union[int, str]],
        return_matrix_as_array: bool = False,
        labels_order: List[Union[int, str]] | None = None,
    ) -> Dict[str, Any]:
        """
        Compute a confusion matrix and simple classification metrics (precision,
        recall, f1, accuracy) given true labels and predicted labels.

        Args:
            true_labels: iterable of ground truth labels
            predicted_labels: iterable of predicted labels

        Returns:
            A dictionary containing:
                - confusion: nested dict mapping true_label -> pred_label -> count
                - per_label: dict mapping label -> {precision, recall, f1}
                - accuracy: float
                - labels: sorted list of labels encountered
        """
        true_list = list(true_labels)
        pred_list = list(predicted_labels)
        if len(true_list) != len(pred_list):
            raise ValueError("true_labels and predicted_labels must have same length")

        labels = list(labels_order) if labels_order is not None else sorted(set(true_list) | set(pred_list))
        # Initialize confusion matrix counts
        confusion: Dict[Union[int, str], Dict[Union[int, str], int]] = {
            t: {p: 0 for p in labels} for t in labels
        }
        for t, p in zip(true_list, pred_list):
            confusion[t][p] += 1

        # Compute metrics per label
        per_label: Dict[Union[int, str], Dict[str, float]] = {}
        total_correct = 0
        total = len(true_list)
        for lbl in labels:
            tp = confusion[lbl][lbl]
            fp = sum(confusion[t][lbl] for t in labels if t != lbl)
            fn = sum(confusion[lbl][p] for p in labels if p != lbl)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            per_label[lbl] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            total_correct += tp

        accuracy = total_correct / total if total > 0 else 0.0

        # Compute micro averages
        total_tp = sum(confusion[lbl][lbl] for lbl in labels)
        total_fp = sum(
            sum(confusion[t][lbl] for t in labels if t != lbl) for lbl in labels
        )
        total_fn = sum(sum(confusion[lbl][p] for p in labels if p != lbl) for lbl in labels)

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )

        # Compute macro averages from per_label metrics
        macro_precision = sum(per_label[l]["precision"] for l in labels) / len(labels) if labels else 0.0
        macro_recall = sum(per_label[l]["recall"] for l in labels) / len(labels) if labels else 0.0
        macro_f1 = sum(per_label[l]["f1"] for l in labels) / len(labels) if labels else 0.0

        response = {
            "confusion": confusion,
            "per_label": per_label,
            "accuracy": accuracy,
            "labels": labels,
            "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
            "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        }

        # Optionally return a 2D confusion matrix array (rows: true, cols: pred) in label order
        if return_matrix_as_array:
            import numpy as _np

            matrix = _np.zeros((len(labels), len(labels)), dtype=int)
            for i, t in enumerate(labels):
                for j, p in enumerate(labels):
                    matrix[i, j] = confusion[t][p]
            # Return as list of lists for portability (JSON friendly)
            response["confusion_matrix"] = matrix.tolist()

        return response

    def evaluate_classification(
        self,
        query_images: List[Union[Image.Image, np.ndarray]] = None,
        true_labels: List[Union[int, str]] = None,
        predicted_labels: List[Union[int, str]] = None,
        n_results: int = 5,
        save_results: bool = False,
        visualizer: Optional[RAGVisualizer] = None,
        output_dir: Optional[str] = None,
        filename_prefix: str = "evaluation",
        to_json: bool = True,
        to_csv: bool = True,
        to_csv_matrix: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate classification either by providing query images and true
        labels (the method will call `batch_classify`) or directly by
        providing predicted_labels and true_labels.
        """
        if predicted_labels is None:
            if query_images is None:
                raise ValueError("Either query_images or predicted_labels must be provided")
            if true_labels is None:
                raise ValueError("true_labels must be provided when evaluating queries")
            batch_results = self.batch_classify(query_images, n_results=n_results)
            predicted_labels = cast(List[Union[int, str]], [r.get("label") for r in batch_results])

        if true_labels is None:
            raise ValueError("true_labels must be provided")

        metrics = self.compute_confusion_and_metrics(true_labels, predicted_labels)

        if save_results:
            # Use provided visualizer if present
            if visualizer is not None:
                viz = visualizer
            else:
                # If an output directory is provided, use it to construct a visualizer
                if output_dir is not None:
                    viz = RAGVisualizer(output_dir=output_dir)
                else:
                    viz = RAGVisualizer(output_dir="./visualizations")
            # Save the results
            saved_paths = viz.save_evaluation_results(
                metrics,
                filename_prefix=filename_prefix,
                to_json=to_json,
                to_csv=to_csv,
                to_csv_matrix=to_csv_matrix,
            )
            # Attach saved_paths to response for convenience
            metrics["saved_paths"] = saved_paths

        return metrics
