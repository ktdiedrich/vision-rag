"""FastAPI service for vision RAG system."""

from typing import List, Optional, Dict, Any
import io
import threading
import traceback
import uvicorn
from contextlib import asynccontextmanager
import json

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from PIL import Image

from .config import (
    CLIP_MODEL_NAME, 
    COLLECTION_NAME, 
    PERSIST_DIRECTORY, 
    MEDMNIST_DATASET, 
    IMAGE_SIZE,
    AVAILABLE_DATASETS,
)
from .encoder import build_encoder
from .rag_store import ChromaRAGStore
from .search import ImageSearcher
from .data_loader import (
    get_human_readable_label,
    get_medmnist_label_names,
    load_medmnist_data,
    get_image_from_array,
)
from .utils import decode_base64_image
from .image_store import ImageFileStore
from .visualization import RAGVisualizer


# Pydantic models for API requests/responses
class SearchRequest(BaseModel):
    """Request model for image search."""
    image_base64: str = Field(..., description="Base64 encoded image")
    n_results: int = Field(5, ge=1, le=100, description="Number of results to return")


class SearchByLabelRequest(BaseModel):
    """Request model for label-based search."""
    label: int = Field(..., ge=0, description="Label to search for")
    n_results: Optional[int] = Field(None, description="Optional limit on results")


class AddImageRequest(BaseModel):
    """Request model for adding an image."""
    image_base64: str = Field(..., description="Base64 encoded image")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Image metadata")


class SearchResult(BaseModel):
    """Search result model."""
    id: str
    distance: float
    metadata: Dict[str, Any]
    human_readable_label: Optional[str] = None


class SearchResponse(BaseModel):
    """Response model for search results."""
    query_info: Dict[str, Any]
    results: List[SearchResult]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    encoder_model: str
    collection_name: str
    embeddings_count: int


class StatsResponse(BaseModel):
    """Statistics response."""
    total_embeddings: int
    total_images: int
    collection_name: str
    persist_directory: str
    image_store_directory: str


class LabelsResponse(BaseModel):
    """Available labels response."""
    labels: Dict[int, str] = Field(..., description="Mapping of label IDs to human-readable names")
    count: int = Field(..., description="Total number of labels")
    dataset: str = Field(..., description="Dataset name")


class DatasetInfo(BaseModel):
    """Information about a MedMNIST dataset."""
    name: str
    description: str
    n_classes: int
    image_size: int
    channels: int


class DatasetsResponse(BaseModel):
    """Available datasets response."""
    datasets: Dict[str, DatasetInfo]
    count: int


class PreloadRequest(BaseModel):
    """Request model for preloading dataset."""
    dataset_name: str = Field(..., description="Name of MedMNIST dataset to load")
    split: str = Field("train", description="Dataset split: 'train', 'test', or 'val'")
    max_images: Optional[int] = Field(None, description="Maximum number of images to load (None for all)")
    size: Optional[int] = Field(None, description="Image size to download (28, 64, 128, or 224). None uses config default.")


class PreloadResponse(BaseModel):
    """Response model for preload operation."""
    status: str
    dataset_name: str
    split: str
    images_loaded: int
    total_embeddings: int
    message: str
    encoder_name: Optional[str] = Field(None, description="Name of the encoder model used for encoding images")


class TsnePlotRequest(BaseModel):
    """Request model for generating t-SNE plot."""
    output_filename: str = Field("tsne_visualization.png", description="Filename for the generated plot")
    method: str = Field("tsne", description="Dimensionality reduction method: 'tsne', 'pca', or 'umap'")
    title: str = Field("RAG Store Embedding Space Visualization", description="Title for the visualization")


class TsnePlotResponse(BaseModel):
    """Response model for t-SNE plot generation."""
    success: bool
    output_path: Optional[str] = None
    total_embeddings: int
    method: str
    unique_labels: Optional[int] = None
    message: str
    error: Optional[str] = None


class ReindexRequest(BaseModel):
    """Request model for reindexing from image store."""
    max_images: Optional[int] = Field(None, description="Optional limit on number of images to process")
    clear_existing: bool = Field(False, description="If True, clear existing embeddings before reindexing")


class ReindexResponse(BaseModel):
    """Response model for reindex operation."""
    success: bool
    images_processed: int
    images_skipped: int
    embeddings_before: int
    total_embeddings: int
    cleared_before_reindex: bool
    message: str
    error: Optional[str] = None


# Global state (initialized on startup)
from .encoder import ImageEncoderProtocol

encoder: Optional[ImageEncoderProtocol] = None
rag_store: Optional[ChromaRAGStore] = None
searcher: Optional[ImageSearcher] = None
image_store: Optional[ImageFileStore] = None

# Thread lock for safe ID generation
_id_generation_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global encoder, rag_store, searcher, image_store
    
    # Startup
    print("🚀 Initializing Vision RAG Service...")
    
    # Initialize encoder
    # Use the encoder factory to create an encoder according to ENCODER_TYPE
    encoder = build_encoder()
    print(f"✅ Loaded encoder: {getattr(encoder, 'model_name', CLIP_MODEL_NAME)} (embedding dim: {encoder.embedding_dimension})")
    
    # Initialize RAG store
    rag_store = ChromaRAGStore(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
    )
    print(f"✅ Connected to ChromaDB ({rag_store.count()} embeddings)")
    
    # Initialize image store with configured image size
    image_store = ImageFileStore(storage_dir="./image_store_api", image_size=IMAGE_SIZE)
    if IMAGE_SIZE:
        print(f"✅ Image store ready ({image_store.count()} images, resize to {IMAGE_SIZE}x{IMAGE_SIZE})")
    else:
        print(f"✅ Image store ready ({image_store.count()} images, no resizing)")
    
    # Initialize searcher
    searcher = ImageSearcher(encoder=encoder, rag_store=rag_store)
    print("✅ Image searcher ready")
    
    print("🎉 Vision RAG Service is ready!")
    
    yield
    
    # Shutdown (cleanup resources if needed)
    print("👋 Shutting down Vision RAG Service...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Vision RAG API",
    description="A retrieval-augmented generation system for medical images using CLIP embeddings",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Vision RAG API",
        "version": "0.1.0",
        "description": "Medical image retrieval using CLIP embeddings",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if encoder is None or rag_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return HealthResponse(
        status="healthy",
        encoder_model=getattr(encoder, "model_name", CLIP_MODEL_NAME),
        collection_name=rag_store.collection_name,
        embeddings_count=rag_store.count(),
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get service statistics."""
    if rag_store is None or image_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return StatsResponse(
        total_embeddings=rag_store.count(),
        total_images=image_store.count(),
        collection_name=rag_store.collection_name,
        persist_directory=rag_store.persist_directory,
        image_store_directory=str(image_store.storage_dir),
    )


@app.get("/labels", response_model=LabelsResponse)
async def get_available_labels():
    """
    Get all available labels from the dataset.
    
    Returns:
        Mapping of label IDs to human-readable names
    """
    try:
        # Get label names for the configured dataset
        label_names = get_medmnist_label_names(dataset_name=MEDMNIST_DATASET)
        
        return LabelsResponse(
            labels=label_names,
            count=len(label_names),
            dataset=MEDMNIST_DATASET,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {str(e)}"
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve labels: {str(e)}"
        )


@app.get("/datasets", response_model=DatasetsResponse)
async def get_available_datasets():
    """
    Get all available MedMNIST datasets that can be loaded.
    
    Returns:
        Information about all available datasets
    """
    datasets_info = {}
    for name, config in AVAILABLE_DATASETS.items():
        datasets_info[name] = DatasetInfo(
            name=name,
            description=config["description"],
            n_classes=config["n_classes"],
            image_size=config["image_size"],
            channels=config["channels"],
        )
    
    return DatasetsResponse(
        datasets=datasets_info,
        count=len(datasets_info),
    )


@app.post("/preload", response_model=PreloadResponse)
async def preload_dataset(request: PreloadRequest):
    """
    Preload a MedMNIST dataset into the RAG store.
    
    This endpoint downloads (if needed) and loads images from a MedMNIST dataset,
    encodes them using CLIP, and stores the embeddings in the RAG store.
    
    Args:
        request: Preload request with dataset name and options
        
    Returns:
        Status of the preload operation
    """
    if encoder is None or rag_store is None or image_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Validate dataset name
    if request.dataset_name not in AVAILABLE_DATASETS:
        available = list(AVAILABLE_DATASETS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dataset '{request.dataset_name}'. Available: {available}"
        )
    
    # Validate split
    if request.split not in ["train", "test", "val"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid split '{request.split}'. Must be 'train', 'test', or 'val'"
        )
    
    try:
        print(f"🔄 Preloading {request.dataset_name} ({request.split} split)...")
        
        # Load dataset
        images, labels = load_medmnist_data(
            dataset_name=request.dataset_name,
            split=request.split,
            size=request.size,
        )
        
        print(f"📦 Loaded {len(images)} images from {request.dataset_name}")
        
        # Limit number of images if specified
        if request.max_images is not None and request.max_images < len(images):
            images = images[:request.max_images]
            labels = labels[:request.max_images]
            print(f"✂️  Limited to {request.max_images} images")
        
        # Convert images to PIL and save to disk
        pil_images = []
        image_paths = []
        for img_array in images:
            pil_img = get_image_from_array(img_array)
            pil_images.append(pil_img)
            
            # Save to image store
            img_path = image_store.save_image(pil_img)
            image_paths.append(img_path)
        
        print(f"💾 Saved {len(pil_images)} images to disk")
        
        # Encode all images
        print(f"🧠 Encoding images with CLIP...")
        embeddings = encoder.encode_images(pil_images)
        print(f"✅ Encoded {len(embeddings)} images")
        
        # Use lock to ensure atomic ID generation and storage
        with _id_generation_lock:
            # Generate IDs
            current_count = rag_store.count()
            ids = [f"{request.dataset_name.lower()}_{request.split}_{current_count + i}" for i in range(len(images))]
            
            # Create metadata with labels and image paths
            metadatas = []
            for i in range(len(images)):
                # Handle both scalar and array labels
                label_value = labels[i]
                if hasattr(label_value, '__len__') and not isinstance(label_value, str):
                    # Multi-dimensional label (e.g., multi-label classification)
                    # Convert to JSON string since ChromaDB doesn't support list metadata
                    label_list = label_value.tolist() if hasattr(label_value, 'tolist') else list(label_value)
                    label_value = json.dumps(label_list)
                else:
                    # Scalar label - keep as int
                    label_value = int(label_value)
                
                metadatas.append({
                    "dataset": request.dataset_name,
                    "split": request.split,
                    "label": label_value,
                    "index": i,
                    "image_path": image_paths[i],
                })
            
            # Add to RAG store
            print(f"📊 Adding {len(embeddings)} embeddings to RAG store...")
            rag_store.add_embeddings(
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )
        
        total_embeddings = rag_store.count()
        message = f"Successfully loaded {len(images)} images from {request.dataset_name} ({request.split})"
        
        print(f"✅ {message}")
        print(f"📊 Total embeddings in store: {total_embeddings}")
        
        return PreloadResponse(
            status="success",
            dataset_name=request.dataset_name,
            split=request.split,
            images_loaded=len(images),
            total_embeddings=total_embeddings,
            encoder_name=getattr(encoder, 'model_name', 'unknown'),
            message=message,
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file not found: {str(e)}"
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error preloading dataset: {str(e)}"
        )


@app.delete("/clear")
async def clear_store(clear_images: bool = False):
    """
    Clear embeddings and optionally images from the RAG store.
    
    Args:
        clear_images: If True, also delete image files from disk (default: False)
    """
    if rag_store is None or image_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    embeddings_before = rag_store.count()
    
    # Clear embeddings
    rag_store.clear()
    
    # Optionally clear images
    images_deleted = 0
    if clear_images:
        images_deleted = image_store.clear()
    
    return {
        "status": "success",
        "embeddings_cleared": embeddings_before,
        "images_deleted": images_deleted,
        "embeddings_remaining": rag_store.count(),
        "images_remaining": image_store.count(),
        "message": f"Cleared {embeddings_before} embeddings" + (f", deleted {images_deleted} images" if clear_images else ""),
    }


@app.post("/search", response_model=SearchResponse)
async def search_similar_images(request: SearchRequest):
    """
    Search for similar images using CLIP embeddings.
    
    Args:
        request: Search request with base64 encoded image
        
    Returns:
        Search results with similar images
    """
    if searcher is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Decode base64 image
        image = decode_base64_image(request.image_base64)
        
        # Perform search
        results = searcher.search(image, n_results=request.n_results)
        
        # Format results
        search_results = []
        for idx, (result_id, distance, metadata) in enumerate(
            zip(results["ids"], results["distances"], results["metadatas"])
        ):
            # Add human readable label if available
            human_label = None
            if "label" in metadata:
                human_label = get_human_readable_label(metadata["label"], dataset_name=MEDMNIST_DATASET)
            
            search_results.append(
                SearchResult(
                    id=result_id,
                    distance=float(distance),
                    metadata=metadata,
                    human_readable_label=human_label,
                )
            )
        
        return SearchResponse(
            query_info={
                "image_size": image.size,
                "image_mode": image.mode,
                "n_results_requested": request.n_results,
            },
            results=search_results,
            count=len(search_results),
        )
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/search/label", response_model=SearchResponse)
async def search_by_label(request: SearchByLabelRequest):
    """
    Search for images by label.
    
    Args:
        request: Search request with label
        
    Returns:
        Images with matching label
    """
    if rag_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Search by label
        results = rag_store.search_by_label(
            label=request.label,
            n_results=request.n_results,
        )
        
        # Format results (no distances for label search)
        search_results = []
        for result_id, metadata in zip(results["ids"], results["metadatas"]):
            human_label = get_human_readable_label(request.label, dataset_name=MEDMNIST_DATASET)
            
            search_results.append(
                SearchResult(
                    id=result_id,
                    distance=0.0,  # No distance for label-based search
                    metadata=metadata,
                    human_readable_label=human_label,
                )
            )
        
        return SearchResponse(
            query_info={
                "label": request.label,
                "human_readable_label": get_human_readable_label(request.label, dataset_name=MEDMNIST_DATASET),
                "n_results_requested": request.n_results,
            },
            results=search_results,
            count=len(search_results),
        )
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error searching by label: {str(e)}")


@app.post("/add")
async def add_image(request: AddImageRequest):
    """
    Add an image to the RAG store.
    
    Args:
        request: Request with base64 encoded image and metadata
        
    Returns:
        Status and assigned ID
    """
    if encoder is None or rag_store is None or image_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Decode base64 image
        image = decode_base64_image(request.image_base64)
        
        # Save image to disk
        image_path = image_store.save_image(image)
        
        # Encode image
        embedding = encoder.encode_image(image)
        
        # Use lock to ensure atomic ID generation and storage
        with _id_generation_lock:
            # Generate ID
            current_count = rag_store.count()
            image_id = f"img_{current_count}"
            
            # Prepare metadata with image path
            metadata = request.metadata.copy() if request.metadata else {}
            metadata["image_path"] = image_path
            metadata["index"] = current_count
            
            # Add to store
            rag_store.add_embeddings(
                embeddings=embedding.reshape(1, -1),
                ids=[image_id],
                metadatas=[metadata],
            )
        
        return {
            "status": "success",
            "id": image_id,
            "metadata": metadata,
            "image_path": image_path,
            "total_embeddings": rag_store.count(),
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error adding image: {str(e)}")


@app.post("/add/batch")
async def add_images_batch(files: List[UploadFile] = File(...)):
    """
    Add multiple images to the RAG store.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        Status and assigned IDs
    """
    if encoder is None or rag_store is None or image_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        images = []
        image_paths = []
        
        for file in files:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
            
            # Save each image to disk
            image_path = image_store.save_image(image)
            image_paths.append(image_path)
        
        # Encode all images
        embeddings = encoder.encode_images(images)
        
        # Use lock to ensure atomic ID generation and storage
        with _id_generation_lock:
            # Generate IDs
            current_count = rag_store.count()
            ids = [f"img_{current_count + i}" for i in range(len(images))]
            
            # Create metadata with image paths
            metadatas = [
                {"index": current_count + i, "image_path": image_paths[i]}
                for i in range(len(images))
            ]
            
            # Add to store
            rag_store.add_embeddings(
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )
        
        return {
            "status": "success",
            "count": len(ids),
            "ids": ids,
            "total_embeddings": rag_store.count(),
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error adding images: {str(e)}")


@app.post("/visualize/tsne", response_model=TsnePlotResponse)
async def generate_tsne_plot(request: TsnePlotRequest):
    """
    Generate a t-SNE visualization of all embeddings in the RAG store.
    
    Creates a 2D scatter plot showing how images cluster in the embedding space.
    The plot is saved to the current directory and the path is returned.
    
    Args:
        request: Request with output filename, method, and title
        
    Returns:
        Status and path to the generated visualization
    """
    if rag_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Get all embeddings and metadata
        all_data = rag_store.get_all_embeddings()
        
        if len(all_data.get("embeddings", [])) == 0:
            return TsnePlotResponse(
                success=False,
                total_embeddings=0,
                method=request.method,
                message="No embeddings found in RAG store. Please add images first.",
                error="No embeddings available",
            )
        
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
        
        # Create visualizer and generate plot
        visualizer = RAGVisualizer(output_dir="./")
        output_path = visualizer.save_embedding_space_visualization(
            embeddings=embeddings,
            labels=labels,
            method=request.method,
            filename=request.output_filename,
                title=request.title,
                model_name=getattr(encoder, "model_name", CLIP_MODEL_NAME),
        )
        
        return TsnePlotResponse(
            success=True,
            output_path=output_path,
            total_embeddings=len(embeddings),
            method=request.method,
            unique_labels=len(set(labels)),
            message=f"Successfully generated {request.method.upper()} plot with {len(embeddings)} embeddings",
        )
        
    except Exception as e:
        traceback.print_exc()
        return TsnePlotResponse(
            success=False,
            total_embeddings=rag_store.count(),
            method=request.method,
            message="Failed to generate visualization",
            error=str(e),
        )


@app.post("/reindex", response_model=ReindexResponse)
async def reindex_from_images(request: ReindexRequest):
    """
    Re-encode all images from the image store and rebuild embeddings.
    
    This is useful when you have images on disk but no embeddings in ChromaDB,
    or when you want to rebuild the index with a different encoder.
    
    Args:
        request: Request with max_images limit and clear_existing flag
        
    Returns:
        Status and statistics of the reindex operation
    """
    if encoder is None or rag_store is None or image_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Get all image files
        image_files = list(image_store.storage_dir.glob("*.png")) + \
                     list(image_store.storage_dir.glob("*.jpg"))
        
        if not image_files:
            return ReindexResponse(
                success=False,
                images_processed=0,
                images_skipped=0,
                embeddings_before=rag_store.count(),
                total_embeddings=rag_store.count(),
                cleared_before_reindex=False,
                message="No images found in image store",
                error=f"No images found in {image_store.storage_dir}",
            )
        
        # Limit if specified
        total_found = len(image_files)
        if request.max_images is not None and request.max_images < len(image_files):
            image_files = image_files[:request.max_images]
        
        # Clear existing embeddings if requested
        embeddings_before = rag_store.count()
        if request.clear_existing:
            rag_store.clear()
        
        # Load and encode images
        pil_images = []
        valid_paths = []
        
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                pil_images.append(img)
                valid_paths.append(str(img_path))
            except Exception as e:
                traceback.print_exc(); continue
        
        if not pil_images:
            return ReindexResponse(
                success=False,
                images_processed=0,
                images_skipped=len(image_files),
                embeddings_before=embeddings_before,
                total_embeddings=rag_store.count(),
                cleared_before_reindex=request.clear_existing,
                message="No valid images could be loaded",
                error="All images failed to load",
            )
        
        # Encode all images
        embeddings = encoder.encode_images(pil_images)
        
        # Use lock for thread-safe ID generation and storage
        with _id_generation_lock:
            current_count = rag_store.count()
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
            rag_store.add_embeddings(
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )
        
        total_embeddings = rag_store.count()
        images_skipped = len(image_files) - len(pil_images)
        
        return ReindexResponse(
            success=True,
            images_processed=len(pil_images),
            images_skipped=images_skipped,
            embeddings_before=embeddings_before,
            total_embeddings=total_embeddings,
            cleared_before_reindex=request.clear_existing,
            message=f"Successfully reindexed {len(pil_images)} images from {total_found} found",
        )
        
    except Exception as e:
        traceback.print_exc()
        return ReindexResponse(
            success=False,
            images_processed=0,
            images_skipped=0,
            embeddings_before=rag_store.count() if rag_store else 0,
            total_embeddings=rag_store.count() if rag_store else 0,
            cleared_before_reindex=request.clear_existing,
            message="Failed to reindex images",
            error=str(e),
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
