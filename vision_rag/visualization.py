"""Visualization utilities for the vision RAG system."""

from pathlib import Path
from typing import List, Union, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from matplotlib.colors import ListedColormap, BoundaryNorm

from .data_loader import get_human_readable_label
import json
import csv


# Maximum number of labels to show on colorbar before omitting tick labels
MAX_COLORBAR_LABELS = 11
from vision_rag.config import MEDMNIST_DATASET  


class RAGVisualizer:
    """Visualizer for RAG system inputs and outputs."""
    
    def __init__(self, output_dir: str = "./visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def save_input_images_grid(
        self, 
        images: List[Union[np.ndarray, Image.Image]], 
        labels: Optional[List[int]] = None,
        filename: str = "input_images_to_rag.png",
        title: str = "Sample Input Images Added to RAG Store",
        max_images: int = 20,
        grid_size: tuple = (4, 5)
    ) -> str:
        """
        Save a grid of sample input images being added to the RAG store.
        
        Args:
            images: List of images (numpy arrays or PIL Images)
            labels: Optional list of labels for the images
            filename: Output filename
            title: Title for the visualization
            max_images: Maximum number of images to display
            grid_size: Grid layout (rows, cols)
            
        Returns:
            Path to saved image
        """
        # Limit number of images
        n_images = min(len(images), max_images, grid_size[0] * grid_size[1])
        
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i in range(grid_size[0] * grid_size[1]):
            row, col = i // grid_size[1], i % grid_size[1]
            ax = axes[row, col] if grid_size[0] > 1 else axes[col]
            
            if i < n_images:
                # Convert to displayable format
                img = images[i]
                if isinstance(img, np.ndarray):
                    if img.ndim == 2:  # Grayscale
                        ax.imshow(img, cmap='gray')
                    else:  # RGB
                        ax.imshow(img)
                else:  # PIL Image
                    ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
                
                # Add label if available
                if labels is not None:
                    human_label = get_human_readable_label(labels[i], dataset_name=MEDMNIST_DATASET)
                    ax.set_title(f'{human_label} ({labels[i]})', fontsize=10)
            else:
                ax.axis('off')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def save_search_input_images(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        labels: Optional[List[int]] = None,
        filename: str = "search_input_images.png",
        title: str = "Search Query Images",
        max_images: int = 10
    ) -> str:
        """
        Save sample search input images.
        
        Args:
            images: List of search query images
            labels: Optional list of labels for the images
            filename: Output filename
            title: Title for the visualization
            max_images: Maximum number of images to display
            
        Returns:
            Path to saved image
        """
        n_images = min(len(images), max_images)
        cols = min(5, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()
        
        for i in range(rows * cols):
            ax = axes[i]
            
            if i < n_images:
                img = images[i]
                if isinstance(img, np.ndarray):
                    if img.ndim == 2:  # Grayscale
                        ax.imshow(img, cmap='gray')
                    else:  # RGB
                        ax.imshow(img)
                else:  # PIL Image
                    ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
                
                if labels is not None:
                    human_label = get_human_readable_label(labels[i], dataset_name=MEDMNIST_DATASET)
                    ax.set_title(f'Query {i+1}\n{human_label} ({labels[i]})', fontsize=10)
                else:
                    ax.set_title(f'Query {i+1}', fontsize=10)
            else:
                ax.axis('off')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def save_search_results(
        self,
        query_image: Union[np.ndarray, Image.Image],
        query_label: Optional[int],
        retrieved_images: List[Union[np.ndarray, Image.Image]],
        retrieved_metadata: List[Dict[str, Any]],
        distances: List[float],
        filename: str = "search_results.png",
        title: str = "RAG Search Results"
    ) -> str:
        """
        Save search results showing query image and retrieved images.
        
        Args:
            query_image: The search query image
            query_label: Optional label for query image
            retrieved_images: List of retrieved images
            retrieved_metadata: List of metadata for retrieved images
            distances: List of distances/similarities
            filename: Output filename
            title: Title for the visualization
            
        Returns:
            Path to saved image
        """
        n_results = len(retrieved_images)
        fig, axes = plt.subplots(1, n_results + 1, figsize=(2*(n_results + 1), 3))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        if n_results == 0:
            axes = [axes]
        
        # Display query image
        ax = axes[0]
        if isinstance(query_image, np.ndarray):
            if query_image.ndim == 2:
                ax.imshow(query_image, cmap='gray')
            else:
                ax.imshow(query_image)
        else:
            ax.imshow(query_image, cmap='gray' if query_image.mode == 'L' else None)
        
        query_title = "Query"
        if query_label is not None:
            human_label = get_human_readable_label(query_label, dataset_name=MEDMNIST_DATASET)
            query_title += f"\n{human_label} ({query_label})"
        ax.set_title(query_title, fontsize=10, fontweight='bold', color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        width = query_image.shape[1] if isinstance(query_image, np.ndarray) else query_image.width
        height = query_image.shape[0] if isinstance(query_image, np.ndarray) else query_image.height
        ax.add_patch(plt.Rectangle((0, 0), width, height, fill=False, edgecolor='red', linewidth=3))
        
        # Display retrieved images
        for i, (img, metadata, dist) in enumerate(zip(retrieved_images, retrieved_metadata, distances)):
            ax = axes[i + 1]
            
            if isinstance(img, np.ndarray):
                if img.ndim == 2:
                    ax.imshow(img, cmap='gray')
                else:
                    ax.imshow(img)
            else:
                ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
            
            # Create title with metadata
            result_title = f"Result {i+1}"
            if 'label' in metadata:
                human_label = get_human_readable_label(metadata['label'], dataset_name=MEDMNIST_DATASET)
                result_title += f"\n{human_label} ({metadata['label']})"
            result_title += f"\nDist: {dist:.3f}"
            
            ax.set_title(result_title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def save_label_distribution(
        self,
        labels: List[int],
        filename: str = "label_distribution.png",
        title: str = "Label Distribution in RAG Store"
    ) -> str:
        """
        Save a visualization of label distribution.
        
        Args:
            labels: List of labels
            filename: Output filename
            title: Title for the visualization
            
        Returns:
            Path to saved image
        """
        df = pd.DataFrame({'label': labels})
        label_counts = df['label'].value_counts().sort_index()

        plt.figure(figsize=(10, 6))
        bars = plt.bar(label_counts.index, label_counts.values, alpha=0.8)
        # Use human readable labels for the x-axis
        tick_labels = [f"{get_human_readable_label(int(lbl), dataset_name=MEDMNIST_DATASET)} ({int(lbl)})" for lbl in label_counts.index]
        plt.xticks(label_counts.index, tick_labels, rotation=45, ha='right')
        plt.xlabel('Label (organ)')
        plt.ylabel('Count')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom')

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return str(output_path)
    
    def save_embedding_space_visualization(
        self,
        embeddings: np.ndarray,
        labels: List[int],
        method: str = 'tsne',
        filename: str = "embedding_space.png",
        title: str = "Embedding Space Visualization",
        model_name: Optional[str] = None,
    ) -> str:
        """
        Save a 2D visualization of the embedding space using dimensionality reduction.
        
        Args:
            embeddings: Array of embeddings (N, embedding_dim)
            labels: List of labels
            method: Dimensionality reduction method ('tsne', 'pca', or 'umap')
            filename: Output filename
            title: Title for the visualization
            
        Returns:
            Path to saved image
        """
        # Perform dimensionality reduction
        if method == 'tsne':
            # Adjust perplexity for small datasets
            perplexity = min(30, max(5, (embeddings.shape[0] - 1) // 3))
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embedding_2d = reducer.fit_transform(embeddings)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings)
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                embedding_2d = reducer.fit_transform(embeddings)
            except ImportError:
                print("UMAP not available, falling back to PCA")
                reducer = PCA(n_components=2, random_state=42)
                embedding_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))

        unique_labels = sorted(list(set(labels)))
        num_labels = len(unique_labels)
        if num_labels == 0:
            raise ValueError("No labels provided for embedding visualization")

        # Map labels to consecutive indices so we can create a discrete colormap
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        mapped_labels = [label_to_idx[l] for l in labels]

        # Build a colormap with as many distinct colors as unique labels
        try:
            # Use seaborn palette for many colors
            palette = sns.color_palette("husl", n_colors=num_labels)
        except Exception:
            # Fallback to matplotlib's tab20: compute `num_labels` colors from the colormap
            cmap = plt.get_cmap('tab20')
            if num_labels == 1:
                palette = [cmap(0.0)]
            else:
                palette = [cmap(i / (num_labels - 1)) for i in range(num_labels)]

        cmap = ListedColormap(palette)
        norm = BoundaryNorm(boundaries=list(range(num_labels + 1)), ncolors=num_labels)

        scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                              c=mapped_labels, cmap=cmap, norm=norm, alpha=0.7)

        # Create a discrete colorbar with human-readable labels
        cbar = plt.colorbar(scatter, ticks=list(range(num_labels)), label='Label')
        readable_labels = [get_human_readable_label(label, dataset_name=MEDMNIST_DATASET) for label in unique_labels]
        cbar.set_ticklabels(readable_labels)
        
        plt.xlabel(f'{method.upper()} 1')
        plt.ylabel(f'{method.upper()} 2')
        # Add model name to the title if available
        full_title = f"{title} ({method.upper()})"
        if model_name:
            full_title = f"{full_title} — Model: {model_name}"
        plt.title(full_title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)

    def plot_confusion_matrix(
        self,
        confusion: Union[Dict[Any, Dict[Any, int]], List[List[int]]],
        labels: Optional[List[Any]] = None,
        label_names: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        normalize: bool = False,
        filename: str = "confusion_matrix.png",
        title: str = "Confusion Matrix",
        cmap: str = "Blues",
        annotate: bool = True,
    ) -> str:
        """
        Plot a confusion matrix as a heatmap and save to a file.

        Args:
            confusion: Nested dict (true_label -> pred_label -> count) or 2D list
            labels: Order of labels corresponding to the matrix rows/columns; required when `confusion` is a 2D list
            normalize: Whether to normalize rows to proportions
            filename: Output filename
            title: Plot title
            cmap: Color map for the heatmap
            annotate: Whether to annotate cells with numbers

        Returns:
            Path to saved image
        """
        # Determine dataset_name for human readable mapping
        ds_name = dataset_name if dataset_name is not None else MEDMNIST_DATASET

        # Convert nested dict to matrix if needed
        if isinstance(confusion, list):
            if labels is None:
                raise ValueError("labels must be provided when passing a 2D list for confusion")
            matrix = np.array(confusion, dtype=float)
            label_order = list(labels)
        else:
            # If confusion is a nested dict, derive ordered labels unless provided
            label_order = list(labels) if labels is not None else list(confusion.keys())
            matrix = np.zeros((len(label_order), len(label_order)), dtype=float)
            for i, t in enumerate(label_order):
                row = confusion.get(t, {})
                for j, p in enumerate(label_order):
                    matrix[i, j] = row.get(p, 0)

        # Create human-readable tick labels
        if label_names is not None:
            tick_labels = list(label_names)
        else:
            # If labels are numeric, get human-readable names for the dataset
            tick_labels = []
            for lab in label_order:
                if isinstance(lab, int):
                    try:
                        tick_labels.append(get_human_readable_label(int(lab), dataset_name=ds_name))
                    except Exception:
                        tick_labels.append(str(lab))
                else:
                    tick_labels.append(str(lab))

        # Optionally normalize rows
        if normalize:
            with np.errstate(all='ignore'):
                row_sums = matrix.sum(axis=1, keepdims=True)
                # Avoid division by zero
                row_sums[row_sums == 0] = 1.0
                norm_matrix = matrix / row_sums
            display_matrix = norm_matrix
            fmt = ".2f"
        else:
            # Ensure integer representation for raw counts so formatting 'd' works
            if not np.issubdtype(matrix.dtype, np.integer):
                try:
                    display_matrix = matrix.astype(int)
                except Exception:
                    # Fallback to using float matrix with float format
                    display_matrix = matrix
                    fmt = ".0f"
                else:
                    fmt = "d"
            else:
                display_matrix = matrix
                fmt = "d"

        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(display_matrix, annot=annotate, fmt=fmt, cmap=cmap,
                 xticklabels=tick_labels, yticklabels=tick_labels, cbar=True)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return str(output_path)

    def plot_per_label_metrics(
        self,
        per_label: Dict[Union[int, str], Dict[str, float]],
        labels_order: Optional[List[Union[int, str]]] = None,
        filename: str = "per_label_metrics.png",
        title: str = "Per-label Precision/Recall/F1",
        figsize: tuple = (10, 6),
        annotate: bool = True,
    ) -> str:
        """
        Plot per-label precision, recall, and F1 in a grouped bar chart.

        Args:
            per_label: dict mapping label -> {precision, recall, f1}
            labels_order: optional list defining the order of labels
            filename: output filename
            title: figure title
            figsize: figure size

        Returns:
            Path to saved image
        """
        import pandas as _pd

        label_order = labels_order if labels_order is not None else list(per_label.keys())

        # Convert to DataFrame
        rows = []
        for lbl in label_order:
            vals = per_label.get(lbl, {})
            rows.append({"label": str(lbl), "precision": vals.get("precision", 0.0), "recall": vals.get("recall", 0.0), "f1": vals.get("f1", 0.0)})
        df = _pd.DataFrame(rows)

        # Melt for grouped bar plot
        df_melt = df.melt(id_vars=["label"], value_vars=["precision", "recall", "f1"], var_name="metric", value_name="value")

        plt.figure(figsize=figsize)
        ax = sns.barplot(data=df_melt, x="label", y="value", hue="metric", palette="muted")
        plt.xlabel("Label")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.0)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(title="Metric")
        # Replace x tick labels with human readable names if available
        ds_name = MEDMNIST_DATASET
        try:
            tick_names = [get_human_readable_label(int(lbl), dataset_name=ds_name) if lbl.isdigit() else lbl for lbl in df["label"].astype(str)]
        except Exception:
            # Fallback to original labels
            tick_names = df["label"].astype(str).tolist()
        ax.set_xticks(list(range(len(tick_names))))
        ax.set_xticklabels(tick_names, rotation=45, ha='right')
        plt.tight_layout()
        # Optionally annotate bar values
        if annotate:
            for p in ax.patches:
                # p is a Rectangle patch; get its height and annotate
                height = p.get_height()
                if np.isfinite(height):
                    ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom', fontsize=8, color='black', rotation=0)
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return str(output_path)

    def plot_roc_curve(
        self,
        true_labels: List[int],
        scores: List[float],
        pos_label: int = 1,
        filename: str = "roc_curve.png",
        title: str = "ROC Curve",
        figsize: tuple = (8, 6),
    ) -> str:
        """
        Plot ROC curve (binary) and save to file.

        Args:
            true_labels: List of ground truth labels (binary)
            scores: List of prediction scores for the positive class
            pos_label: Label considered positive
            filename: Output filename
            title: Plot title
            figsize: Figure size

        Returns:
            Path to saved image
        """
        y_true = np.array(true_labels)
        y_scores = np.array(scores)

        # If y_true includes multi-class labels, binarize according to pos_label
        if y_true.ndim == 1 and len(np.unique(y_true)) > 2:
            y_true_binary = (y_true == pos_label).astype(int)
        else:
            y_true_binary = y_true

        # For clarity ensure y_scores is a 1-d vector (required by sklearn for binary curves)
        if y_scores.ndim != 1:
            y_scores = y_scores.ravel()

        fpr, tpr, _ = roc_curve(y_true_binary, y_scores, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        return str(out_path)

    def plot_precision_recall_curve(
        self,
        true_labels: List[int],
        scores: List[float],
        pos_label: int = 1,
        filename: str = "pr_curve.png",
        title: str = "Precision-Recall Curve",
        figsize: tuple = (8, 6),
    ) -> str:
        """
        Plot Precision-Recall curve (binary) and save to file.

        Args:
            true_labels: List of ground truth labels (binary)
            scores: List of prediction scores for the positive class
            pos_label: Label considered positive
            filename: Output filename
            title: Plot title
            figsize: Figure size

        Returns:
            Path to saved image
        """
        y_true = np.array(true_labels)
        y_scores = np.array(scores)

        # If y_true includes multi-class labels, binarize according to pos_label
        if y_true.ndim == 1 and len(np.unique(y_true)) > 2:
            y_true_binary = (y_true == pos_label).astype(int)
        else:
            y_true_binary = y_true

        # Ensure y_scores is 1-D (sklearn expects 1D score array for binary precision/recall)
        if y_scores.ndim != 1:
            y_scores = y_scores.ravel()

        precision, recall, _ = precision_recall_curve(y_true_binary, y_scores, pos_label=pos_label)
        ap = average_precision_score(y_true_binary, y_scores)

        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='darkgreen', lw=2, label=f'AP = {ap:.2f}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        return str(out_path)

    def save_evaluation_results(
        self,
        metrics: Dict[str, Any],
        filename_prefix: str = "evaluation",
        to_json: bool = True,
        to_csv: bool = True,
        to_csv_matrix: bool = False,
    ) -> Dict[str, str]:
        """
        Save evaluation metrics and confusion matrix to JSON and CSV files.

        Arguments:
            metrics: the dict returned from `ImageSearcher.compute_confusion_and_metrics`
            filename_prefix: filename prefix used for saved files
            to_json: whether to write a JSON file
            to_csv: whether to write CSV files (confusion + per-class metrics)

        Returns:
            dict mapping file types to saved paths
        """
        saved_paths: Dict[str, str] = {}

        # Ensure the metrics is serializable; convert numpy types if present
        serializable_metrics = json.loads(json.dumps(metrics, default=lambda o: (o.tolist() if hasattr(o, 'tolist') else str(o))))

        if to_json:
            json_path = self.output_dir / f"{filename_prefix}.json"
            with open(json_path, "w") as fh:
                json.dump(serializable_metrics, fh, indent=2)
            saved_paths["json"] = str(json_path)

        # CSV export: confusion and per_class
        if to_csv:
            # Determine confusion structure
            confusion = metrics.get("confusion")
            labels = metrics.get("labels") or []

            # If confusion is nested dict: dump triplets
            if isinstance(confusion, dict):
                conf_csv_path = self.output_dir / f"{filename_prefix}_confusion.csv"
                with open(conf_csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["true_label", "predicted_label", "count"])
                    # Iterate labels order if present, else keys
                    label_order = list(labels) if labels else list(confusion.keys())
                    for t in label_order:
                        row = confusion.get(t, {})
                        for p in label_order:
                            writer.writerow([t, p, row.get(p, 0)])
                saved_paths["confusion_csv"] = str(conf_csv_path)
                if to_csv_matrix:
                    # Also write out matrix header format
                    matrix_csv_path = self.output_dir / f"{filename_prefix}_confusion_matrix.csv"
                    with open(matrix_csv_path, "w", newline="") as mfile:
                        writer = csv.writer(mfile)
                        # Write header: empty cell + predicted labels
                        writer.writerow([""] + [str(p) for p in label_order])
                        for t in label_order:
                            row = confusion.get(t, {})
                            writer.writerow([str(t)] + [row.get(p, 0) for p in label_order])
                    saved_paths["confusion_matrix_csv"] = str(matrix_csv_path)
            elif isinstance(confusion, list):
                # confusion is a 2D list
                if not labels:
                    raise ValueError("labels must be provided when passing matrix list for CSV export")
                conf_csv_path = self.output_dir / f"{filename_prefix}_confusion.csv"
                with open(conf_csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["true_label", "predicted_label", "count"])
                    for i, t in enumerate(labels):
                        row = confusion[i]
                        for j, p in enumerate(labels):
                            writer.writerow([t, p, row[j]])
                saved_paths["confusion_csv"] = str(conf_csv_path)
                if to_csv_matrix:
                    matrix_csv_path = self.output_dir / f"{filename_prefix}_confusion_matrix.csv"
                    with open(matrix_csv_path, "w", newline="") as mfile:
                        writer = csv.writer(mfile)
                        writer.writerow([""] + [str(p) for p in labels])
                        for i, t in enumerate(labels):
                            row = confusion[i]
                            writer.writerow([str(t)] + [row[j] for j in range(len(labels))])
                    saved_paths["confusion_matrix_csv"] = str(matrix_csv_path)

            # Write per-class metrics
            per_label = metrics.get("per_label") or {}
            per_csv_path = self.output_dir / f"{filename_prefix}_per_label.csv"
            with open(per_csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["label", "precision", "recall", "f1"])
                label_order = list(labels) if labels else list(per_label.keys())
                for lbl in label_order:
                    vals = per_label.get(lbl, {})
                    writer.writerow([lbl, vals.get("precision", 0.0), vals.get("recall", 0.0), vals.get("f1", 0.0)])
            saved_paths["per_label_csv"] = str(per_csv_path)

        return saved_paths