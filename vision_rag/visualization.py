"""Visualization utilities for the vision RAG system."""

from pathlib import Path
from typing import List, Union, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd


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
                    ax.set_title(f'Label: {labels[i]}', fontsize=10)
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
                    ax.set_title(f'Query {i+1}\nLabel: {labels[i]}', fontsize=10)
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
            query_title += f"\nLabel: {query_label}"
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
                result_title += f"\nLabel: {metadata['label']}"
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
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
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
        title: str = "Embedding Space Visualization"
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
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
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
        scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                            c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, label='Label')
        plt.xlabel(f'{method.upper()} 1')
        plt.ylabel(f'{method.upper()} 2')
        plt.title(f'{title} ({method.upper()})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)