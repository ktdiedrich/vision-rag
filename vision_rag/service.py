"""FastAPI service for vision RAG system."""

from typing import List, Optional, Dict, Any
import io
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from PIL import Image

from .config import CLIP_MODEL_NAME, COLLECTION_NAME, PERSIST_DIRECTORY, MEDMNIST_DATASET, IMAGE_SIZE
from .encoder import CLIPImageEncoder
from .rag_store import ChromaRAGStore
from .search import ImageSearcher
from .data_loader import get_human_readable_label
from .utils import decode_base64_image
from .image_store import ImageFileStore
from .data_loader import get_medmnist_label_names


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


# Global state (initialized on startup)
encoder: Optional[CLIPImageEncoder] = None
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
    encoder = CLIPImageEncoder(model_name=CLIP_MODEL_NAME)
    print(f"✅ Loaded CLIP encoder: {CLIP_MODEL_NAME} (embedding dim: {encoder.embedding_dimension})")
    
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
        encoder_model=CLIP_MODEL_NAME,
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
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve labels: {str(e)}"
        )


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
        raise HTTPException(status_code=400, detail=f"Error adding images: {str(e)}")


@app.delete("/clear")
async def clear_store():
    """Clear all embeddings from the RAG store."""
    if rag_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    rag_store.clear()
    
    return {
        "status": "success",
        "message": "All embeddings cleared",
        "total_embeddings": rag_store.count(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
