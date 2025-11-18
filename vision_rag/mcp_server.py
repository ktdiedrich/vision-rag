"""Model Context Protocol (MCP) server for vision RAG agent communication."""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional, Callable, Coroutine
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import traceback

from .config import (
    CLIP_MODEL_NAME, 
    MEDMNIST_DATASET, 
    IMAGE_SIZE,
    AVAILABLE_DATASETS,
)
from .encoder import CLIPImageEncoder
from .rag_store import ChromaRAGStore
from .search import ImageSearcher
from .data_loader import (
    get_human_readable_label,
    get_medmnist_label_names,
    load_medmnist_data,
    get_image_from_array,
)
from .utils import decode_base64_image, encode_image_to_base64
from .image_store import ImageFileStore
from PIL import Image
import numpy as np
from .visualization import RAGVisualizer


class VisionRAGMCPServer:
    """MCP server for vision RAG system enabling agent-to-agent communication."""
    
    def __init__(
        self,
        encoder_model: str = CLIP_MODEL_NAME,
        collection_name: str = "mcp_vision_rag",
        persist_directory: str = "./chroma_db_mcp",
        image_store_dir: str = "./image_store_mcp",
        image_size: Optional[int] = IMAGE_SIZE,
    ):
        """
        Initialize MCP server.
        
        Args:
            encoder_model: CLIP model name
            collection_name: ChromaDB collection name
            persist_directory: Directory for persistent storage
            image_store_dir: Directory for image file storage
            image_size: Target size for storing images (width and height).
                       If provided, images will be resized to (image_size, image_size).
                       If None, images are stored at original size.
        """
        self.encoder = CLIPImageEncoder(model_name=encoder_model)
        self.rag_store = ChromaRAGStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        self.image_store = ImageFileStore(storage_dir=image_store_dir, image_size=image_size)
        self.searcher = ImageSearcher(encoder=self.encoder, rag_store=self.rag_store)
        
        # MCP protocol handlers
        # Keys map to async tool callables which return a coroutine
        self.tools: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {
            "search_similar_images": self.search_similar_images,
            "search_by_label": self.search_by_label,
            "add_image": self.add_image,
            "get_statistics": self.get_statistics,
            "list_available_labels": self.list_available_labels,
            "list_available_datasets": self.list_available_datasets,
            "preload_dataset": self.preload_dataset,
            "clear_store": self.clear_store,
            "reindex_from_images": self.reindex_from_images,
            "generate_tsne_plot": self.generate_tsne_plot,
        }
    
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming tool calls from other agents.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            return {
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self.tools.keys()),
            }
        
        try:
            result = await self.tools[tool_name](**arguments)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def search_similar_images(
        self,
        image_base64: str,
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Search for similar images.
        
        Args:
            image_base64: Base64 encoded image
            n_results: Number of results to return
            
        Returns:
            Search results with metadata
        """
        # Decode image
        image = decode_base64_image(image_base64)
        
        # Perform search
        results = self.searcher.search(image, n_results=n_results)
        
        # Enrich results with human-readable labels
        enriched_results = []
        for result_id, distance, metadata in zip(
            results["ids"], results["distances"], results["metadatas"]
        ):
            result_dict = {
                "id": result_id,
                "distance": float(distance),
                "metadata": metadata,
            }
            
            # Add human-readable label if available
            if "label" in metadata:
                result_dict["human_readable_label"] = get_human_readable_label(metadata["label"], dataset_name=MEDMNIST_DATASET)
            
            enriched_results.append(result_dict)
        
        return {
            "query_info": {
                "image_size": list(image.size),
                "image_mode": image.mode,
            },
            "results": enriched_results,
            "count": len(enriched_results),
        }
    
    async def search_by_label(
        self,
        label: int,
        n_results: Optional[int] = None,
        return_images: bool = False,
    ) -> Dict[str, Any]:
        """
        Search for images by label.
        
        Args:
            label: Label to search for
            n_results: Optional limit on results
            return_images: If True, return base64 encoded images
            
        Returns:
            Images with matching label, optionally including base64 encoded image data
        """
        results = self.rag_store.search_by_label(label=label, n_results=n_results)
        
        response = {
            "label": label,
            "human_readable_label": get_human_readable_label(label, dataset_name=MEDMNIST_DATASET),
            "ids": results["ids"],
            "metadatas": results["metadatas"],
            "count": len(results["ids"]),
        }
        
        # Optionally load and encode images
        if return_images:
            images: List[Optional[str]] = []
            for metadata in results["metadatas"]:
                if "image_path" in metadata:
                    image_path = Path(metadata["image_path"])
                    if image_path.exists():
                        try:
                            img = Image.open(image_path)
                            images.append(encode_image_to_base64(img))
                        except Exception:
                            # Handle corrupted or unopenable images
                            images.append(None)
                    else:
                        images.append(None)
                else:
                    images.append(None)
            response["images_base64"] = images
        
        return response
    
    async def add_image(
        self,
        image_base64: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add an image to the RAG store.
        
        Args:
            image_base64: Base64 encoded image
            metadata: Optional metadata
            
        Returns:
            Status and assigned ID
        """
        # Decode image
        image = decode_base64_image(image_base64)
        
        # Save image to disk
        image_path = self.image_store.save_image(image)
        
        # Encode image
        embedding = self.encoder.encode_image(image)
        
        # Generate ID
        current_count = self.rag_store.count()
        image_id = f"img_{current_count}"
        
        # Prepare metadata with image path
        if metadata is None:
            metadata = {}
        metadata["image_path"] = image_path
        metadata["index"] = current_count
        
        # Add to store
        self.rag_store.add_embeddings(
            embeddings=embedding.reshape(1, -1),
            ids=[image_id],
            metadatas=[metadata],
        )
        
        return {
            "id": image_id,
            "metadata": metadata,
            "image_path": image_path,
            "total_embeddings": self.rag_store.count(),
        }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get RAG store statistics.
        
        Returns:
            Statistics about the store
        """
        # Get absolute paths for debugging
        image_store_abs = self.image_store.storage_dir.resolve()
        chroma_abs = Path(self.rag_store.persist_directory).resolve()
        
        return {
            "total_embeddings": self.rag_store.count(),
            "total_images": self.image_store.count(),
            "collection_name": self.rag_store.collection_name,
            "persist_directory": self.rag_store.persist_directory,
            "persist_directory_absolute": str(chroma_abs),
            "image_store_directory": str(self.image_store.storage_dir),
            "image_store_directory_absolute": str(image_store_abs),
            "current_working_directory": str(Path.cwd()),
            "encoder_model": CLIP_MODEL_NAME,
            "embedding_dimension": self.encoder.embedding_dimension,
        }
    
    async def list_available_labels(self) -> Dict[str, Any]:
        """
        List all available organ labels.
        
        Returns:
            Mapping of label IDs to human-readable names
        """
        label_names = get_medmnist_label_names(dataset_name="OrganSMNIST")
        
        return {
            "labels": label_names,
            "count": len(label_names),
        }
    
    async def list_available_datasets(self) -> Dict[str, Any]:
        """
        List all available MedMNIST datasets that can be loaded.
        
        Returns:
            Information about available datasets
        """
        datasets_info = {}
        for name, config in AVAILABLE_DATASETS.items():
            datasets_info[name] = {
                "description": config["description"],
                "n_classes": config["n_classes"],
                "image_size": config["image_size"],
                "channels": config["channels"],
            }
        
        return {
            "datasets": datasets_info,
            "count": len(datasets_info),
        }
    
    async def clear_store(
        self,
        clear_embeddings: bool = True,
        clear_images: bool = False,
    ) -> Dict[str, Any]:
        """
        Clear embeddings and/or images from the RAG store.
        
        Args:
            clear_embeddings: If True, clear all embeddings from ChromaDB (default: True)
            clear_images: If True, also delete all image files from disk (default: False)
            
        Returns:
            Status and counts of cleared items
        """
        try:
            embeddings_before = self.rag_store.count()
            images_before = self.image_store.count()
            
            images_deleted = 0
            if clear_embeddings:
                print(f"🗑️  Clearing {embeddings_before} embeddings from ChromaDB...", file=sys.stderr)
                self.rag_store.clear()
                print(f"✅ Cleared embeddings", file=sys.stderr)
            
            if clear_images:
                print(f"🗑️  Deleting {images_before} image files from disk...", file=sys.stderr)
                images_deleted = self.image_store.clear()
                print(f"✅ Deleted {images_deleted} image files", file=sys.stderr)
            
            embeddings_after = self.rag_store.count()
            images_after = self.image_store.count()
            
            return {
                "success": True,
                "embeddings_cleared": embeddings_before - embeddings_after,
                "images_deleted": images_deleted,
                "embeddings_remaining": embeddings_after,
                "images_remaining": images_after,
                "message": f"Cleared {embeddings_before - embeddings_after} embeddings, deleted {images_deleted} images",
            }
            
        except Exception as e:
            error_msg = f"Error clearing store: {str(e)}"
            print(f"❌ {error_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {
                "success": False,
                "error": error_msg,
            }
    
    async def preload_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        max_images: Optional[int] = None,
        size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Preload a MedMNIST dataset into the RAG store.
        
        Downloads (if needed) and loads images from a MedMNIST dataset,
        encodes them using CLIP, and stores the embeddings in the RAG store.
        
        Args:
            dataset_name: Name of MedMNIST dataset (e.g., 'PathMNIST', 'ChestMNIST')
            split: Dataset split - 'train', 'test', or 'val'
            max_images: Optional maximum number of images to load (None for all)
            size: Image size to download (28, 64, 128, or 224). None uses config default.
            
        Returns:
            Status and statistics of the preload operation
        """
        # Validate dataset name
        if dataset_name not in AVAILABLE_DATASETS:
            available = list(AVAILABLE_DATASETS.keys())
            return {
                "success": False,
                "error": f"Unknown dataset '{dataset_name}'. Available: {available}",
            }
        
        # Validate split
        if split not in ["train", "test", "val"]:
            return {
                "success": False,
                "error": f"Invalid split '{split}'. Must be 'train', 'test', or 'val'",
            }
        
        try:
            print(f"🔄 Preloading {dataset_name} ({split} split)...", file=sys.stderr)
            
            # Load dataset
            images, labels = load_medmnist_data(
                dataset_name=dataset_name,
                split=split,
                size=size,
            )
            
            print(f"📦 Loaded {len(images)} images from {dataset_name}", file=sys.stderr)
            
            # Limit number of images if specified
            if max_images is not None and max_images < len(images):
                images = images[:max_images]
                labels = labels[:max_images]
                print(f"✂️  Limited to {max_images} images", file=sys.stderr)
            
            # Convert images to PIL and save to disk
            pil_images = []
            image_paths = []
            for img_array in images:
                pil_img = get_image_from_array(img_array)
                pil_images.append(pil_img)
                
                # Save to image store
                img_path = self.image_store.save_image(pil_img)
                image_paths.append(img_path)
            
            print(f"💾 Saved {len(pil_images)} images to disk", file=sys.stderr)
            
            # Encode all images
            print(f"🧠 Encoding images with CLIP...", file=sys.stderr)
            embeddings = self.encoder.encode_images(pil_images)
            print(f"✅ Encoded {len(embeddings)} images", file=sys.stderr)
            
            # Generate IDs
            ids = [f"{dataset_name.lower()}_{split}_{i}" for i in range(len(images))]
            
            # Create metadata with labels and image paths
            metadatas = []
            for i in range(len(images)):
                # Handle both scalar and array labels
                label_value = labels[i]
                if hasattr(label_value, '__len__') and not isinstance(label_value, str):
                    # Multi-dimensional label (e.g., multi-label classification)
                    # Convert to JSON string since ChromaDB doesn't support list metadata
                    import json as json_lib
                    label_list = label_value.tolist() if hasattr(label_value, 'tolist') else list(label_value)
                    label_value = json_lib.dumps(label_list)
                else:
                    # Scalar label - keep as int
                    label_value = int(label_value)
                
                metadatas.append({
                    "dataset": dataset_name,
                    "split": split,
                    "label": label_value,
                    "index": i,
                    "image_path": image_paths[i],
                })
            
            # Add to RAG store
            print(f"📊 Adding {len(embeddings)} embeddings to RAG store...", file=sys.stderr)
            self.rag_store.add_embeddings(
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )
            
            total_embeddings = self.rag_store.count()
            message = f"Successfully loaded {len(images)} images from {dataset_name} ({split})"
            
            print(f"✅ {message}", file=sys.stderr)
            print(f"📊 Total embeddings in store: {total_embeddings}", file=sys.stderr)
            
            return {
                "success": True,
                "dataset_name": dataset_name,
                "split": split,
                "images_loaded": len(images),
                "total_embeddings": total_embeddings,
                "message": message,
            }
            
        except Exception as e:
            error_msg = f"Error preloading dataset: {str(e)}"
            print(f"❌ {error_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {
                "success": False,
                "error": error_msg,
            }
    
    async def reindex_from_images(
        self,
        max_images: Optional[int] = None,
        clear_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Re-encode all images from the image store and rebuild the embeddings database.
        
        This is useful when you have images on disk but no embeddings in ChromaDB,
        or when you want to rebuild the index with a different encoder.
        
        Args:
            max_images: Optional limit on number of images to process (None for all)
            clear_existing: If True, clear existing embeddings before reindexing (default: False)
            
        Returns:
            Status and statistics of the reindex operation
        """
        try:
            print(f"🔄 Reindexing images from {self.image_store.storage_dir}...", file=sys.stderr)
            
            # Get all image files
            image_files = list(self.image_store.storage_dir.glob("*.png")) + \
                         list(self.image_store.storage_dir.glob("*.jpg"))
            
            if not image_files:
                return {
                    "success": False,
                    "error": "No images found in image store",
                    "image_store_directory": str(self.image_store.storage_dir),
                }
            
            print(f"📦 Found {len(image_files)} images in image store", file=sys.stderr)
            
            # Limit if specified
            if max_images is not None and max_images < len(image_files):
                image_files = image_files[:max_images]
                print(f"✂️  Limited to {max_images} images", file=sys.stderr)
            
            # Clear existing embeddings if requested
            embeddings_before = self.rag_store.count()
            if clear_existing:
                print(f"🗑️  Clearing {embeddings_before} existing embeddings...", file=sys.stderr)
                self.rag_store.clear()
            
            # Load and encode images
            print(f"🖼️  Loading {len(image_files)} images...", file=sys.stderr)
            pil_images = []
            valid_paths = []
            
            for img_path in image_files:
                try:
                    img = Image.open(img_path)
                    pil_images.append(img)
                    valid_paths.append(str(img_path))
                except Exception as e:
                    print(f"⚠️  Failed to load {img_path.name}: {e}", file=sys.stderr)
                    continue
            
            if not pil_images:
                return {
                    "success": False,
                    "error": "No valid images could be loaded",
                }
            
            print(f"✅ Loaded {len(pil_images)} valid images", file=sys.stderr)
            
            # Encode all images
            print(f"🧠 Encoding images with CLIP...", file=sys.stderr)
            embeddings = self.encoder.encode_images(pil_images)
            print(f"✅ Encoded {len(embeddings)} images", file=sys.stderr)
            
            # Generate IDs and metadata
            current_count = self.rag_store.count()
            ids = [f"reindexed_{current_count + i}" for i in range(len(pil_images))]
            
            metadatas = [
                {
                    "image_path": valid_paths[i],
                    "index": current_count + i,
                    "reindexed": True,
                }
                for i in range(len(pil_images))
            ]
            
            # Add to RAG store
            print(f"📊 Adding {len(embeddings)} embeddings to RAG store...", file=sys.stderr)
            self.rag_store.add_embeddings(
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )
            
            total_embeddings = self.rag_store.count()
            message = f"Successfully reindexed {len(pil_images)} images"
            
            print(f"✅ {message}", file=sys.stderr)
            print(f"📊 Total embeddings in store: {total_embeddings}", file=sys.stderr)
            
            return {
                "success": True,
                "images_processed": len(pil_images),
                "images_skipped": len(image_files) - len(pil_images),
                "embeddings_before": embeddings_before,
                "total_embeddings": total_embeddings,
                "cleared_before_reindex": clear_existing,
                "message": message,
            }
            
        except Exception as e:
            error_msg = f"Error reindexing images: {str(e)}"
            print(f"❌ {error_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {
                "success": False,
                "error": error_msg,
            }
    
    async def generate_tsne_plot(
        self,
        output_filename: str = "tsne_visualization.png",
        method: str = "tsne",
        title: str = "RAG Store Embedding Space Visualization",
    ) -> Dict[str, Any]:
        """
        Generate a t-SNE (or other dimensionality reduction) plot of all embeddings in the RAG store.
        
        Args:
            output_filename: Filename for the generated plot (saved in current directory)
            method: Dimensionality reduction method ('tsne', 'pca', or 'umap')
            title: Title for the visualization
            
        Returns:
            Status and path to the generated plot
        """
        try:
            print(f"📊 Generating {method.upper()} visualization...", file=sys.stderr)
            
            # Get all embeddings and metadata
            all_data = self.rag_store.get_all_embeddings()
            
            if len(all_data.get("embeddings", [])) == 0:
                return {
                    "success": False,
                    "error": "No embeddings found in RAG store. Please add images first.",
                    "total_embeddings": 0,
                }
            
            # Convert embeddings to numpy array
            embeddings = np.array(all_data["embeddings"])
            
            # Extract labels from metadata
            labels = []
            for meta in all_data["metadatas"]:
                # Try to get label from metadata, default to 0 if not found
                label = meta.get("label", meta.get("index", 0))
                # Convert to int, handling various types
                try:
                    if isinstance(label, (list, tuple, np.ndarray)):
                        label = label[0] if len(label) > 0 else 0
                    labels.append(int(label))
                except (ValueError, TypeError):
                    labels.append(0)
            
            print(f"📈 Processing {len(embeddings)} embeddings...", file=sys.stderr)
            
            # Create visualizer and generate plot
            visualizer = RAGVisualizer(output_dir="./")
            output_path = visualizer.save_embedding_space_visualization(
                embeddings=embeddings,
                labels=labels,
                method=method,
                filename=output_filename,
                title=title,
            )
            
            print(f"✅ Saved visualization to: {output_path}", file=sys.stderr)
            
            return {
                "success": True,
                "output_path": output_path,
                "total_embeddings": len(embeddings),
                "method": method,
                "unique_labels": len(set(labels)),
                "message": f"Successfully generated {method.upper()} plot with {len(embeddings)} embeddings",
            }
            
        except Exception as e:
            error_msg = f"Error generating {method.upper()} plot: {str(e)}"
            print(f"❌ {error_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {
                "success": False,
                "error": error_msg,
            }
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get MCP tool definitions for agent discovery.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "name": "search_similar_images",
                "description": "Search for visually similar medical images using CLIP embeddings",
                "parameters": {
                    "image_base64": {
                        "type": "string",
                        "description": "Base64 encoded image to search for",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of similar images to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["image_base64"],
            },
            {
                "name": "search_by_label",
                "description": "Search for medical images by organ label",
                "parameters": {
                    "label": {
                        "type": "integer",
                        "description": "Organ label (0-10: bladder, femur-left, femur-right, heart, kidney-left, kidney-right, liver, lung-left, lung-right, pancreas, spleen)",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Optional limit on number of results",
                    },
                    "return_images": {
                        "type": "boolean",
                        "description": "If True, return base64 encoded images (default: False)",
                        "default": False,
                    },
                },
                "required": ["label"],
            },
            {
                "name": "add_image",
                "description": "Add a medical image to the RAG store",
                "parameters": {
                    "image_base64": {
                        "type": "string",
                        "description": "Base64 encoded image to add",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata (e.g., label, patient_id)",
                    },
                },
                "required": ["image_base64"],
            },
            {
                "name": "get_statistics",
                "description": "Get statistics about the vision RAG store",
                "parameters": {},
                "required": [],
            },
            {
                "name": "list_available_labels",
                "description": "List all available organ labels with human-readable names",
                "parameters": {},
                "required": [],
            },
            {
                "name": "list_available_datasets",
                "description": "List all available MedMNIST datasets that can be preloaded",
                "parameters": {},
                "required": [],
            },
            {
                "name": "preload_dataset",
                "description": "Preload a MedMNIST dataset into the RAG store. Downloads (if needed), encodes with CLIP, and stores embeddings.",
                "parameters": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of MedMNIST dataset (e.g., 'PathMNIST', 'ChestMNIST', 'OrganSMNIST')",
                    },
                    "split": {
                        "type": "string",
                        "description": "Dataset split: 'train', 'test', or 'val' (default: 'train')",
                        "default": "train",
                    },
                    "max_images": {
                        "type": "integer",
                        "description": "Optional maximum number of images to load (None/omit for all images)",
                    },
                    "size": {
                        "type": "integer",
                        "description": "Image size to download: 28, 64, 128, or 224 (None/omit for config default)",
                    },
                },
                "required": ["dataset_name"],
            },
            {
                "name": "clear_store",
                "description": "Clear embeddings and/or images from the RAG store. Use this to reset the database.",
                "parameters": {
                    "clear_embeddings": {
                        "type": "boolean",
                        "description": "If True, clear all embeddings from ChromaDB (default: True)",
                        "default": True,
                    },
                    "clear_images": {
                        "type": "boolean",
                        "description": "If True, also delete all image files from disk (default: False). WARNING: This permanently deletes files!",
                        "default": False,
                    },
                },
                "required": [],
            },
            {
                "name": "reindex_from_images",
                "description": "Re-encode all images from the image store and rebuild embeddings. Useful when you have images but no embeddings.",
                "parameters": {
                    "max_images": {
                        "type": "integer",
                        "description": "Optional maximum number of images to process (None/omit for all images)",
                    },
                    "clear_existing": {
                        "type": "boolean",
                        "description": "If True, clear existing embeddings before reindexing (default: False)",
                        "default": False,
                    },
                },
                "required": [],
            },
            {
                "name": "generate_tsne_plot",
                "description": "Generate a t-SNE (or PCA/UMAP) visualization of all embeddings in the RAG store. Creates a 2D scatter plot showing how images cluster in embedding space.",
                "parameters": {
                    "output_filename": {
                        "type": "string",
                        "description": "Filename for the generated plot (default: 'tsne_visualization.png')",
                        "default": "tsne_visualization.png",
                    },
                    "method": {
                        "type": "string",
                        "description": "Dimensionality reduction method: 'tsne', 'pca', or 'umap' (default: 'tsne')",
                        "default": "tsne",
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the visualization (default: 'RAG Store Embedding Space Visualization')",
                        "default": "RAG Store Embedding Space Visualization",
                    },
                },
                "required": [],
            },
        ]


async def main():
    """Run MCP server using stdio transport."""
    # Create the MCP server instance
    mcp_server = Server("vision-rag")
    
    # Initialize Vision RAG components
    vision_rag = VisionRAGMCPServer()
    
    print("🤖 Vision RAG MCP Server Starting...", file=sys.stderr)
    print(f"📊 Total embeddings: {vision_rag.rag_store.count()}", file=sys.stderr)
    print(f"🔧 Available tools: {list(vision_rag.tools.keys())}", file=sys.stderr)
    
    # Register tools with MCP server
    @mcp_server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available tools."""
        tools = []
        for tool_def in vision_rag.get_tool_definitions():
            # Convert our tool definitions to MCP Tool format with proper JSON Schema
            input_schema = {
                "type": "object",
                "properties": tool_def["parameters"],
            }
            # Add required field if it exists
            if "required" in tool_def and tool_def["required"]:
                input_schema["required"] = tool_def["required"]
            
            tools.append(Tool(
                name=tool_def["name"],
                description=tool_def["description"],
                inputSchema=input_schema
            ))
        return tools
    
    @mcp_server.call_tool()
    async def call_tool(name: str, arguments: dict) -> List[TextContent]:
        """Execute a tool by name."""
        try:
            print(f"🔧 Calling tool: {name} with arguments: {arguments}", file=sys.stderr)
            result = await vision_rag.handle_tool_call(name, arguments)
            
            result_json = json.dumps(result, indent=2)
            print(f"✅ Tool {name} completed successfully", file=sys.stderr)
            return [TextContent(
                type="text",
                text=result_json
            )]
        except Exception as e:
            print(f"❌ Error in tool {name}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise
    
    print("✅ MCP Server ready for agent communication", file=sys.stderr)
    
    # Run the server with stdio transport
    try:
        async with stdio_server() as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options()
            )
    except Exception as e:
        print(f"❌ MCP Server error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise

if __name__ == "__main__":
    asyncio.run(main())
