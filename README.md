# vision-rag

A retrieval-augmented generation (RAG) system for medical images using CLIP embeddings and ChromaDB.

## Overview

This project implements a vision-based RAG system that:
- Downloads medical imaging datasets from the MedMNIST collection (12+ datasets supported)
- Encodes medical images using CLIP models (default: ViT-B-32, configurable)
- Stores embeddings in ChromaDB for efficient retrieval
- Searches for similar images from the training set given test images
- Supports multiple MedMNIST datasets: OrganSMNIST, PathMNIST, ChestMNIST, DermaMNIST, and more

## Configuration

Vision RAG can be configured using environment variables:

```bash
# Choose which MedMNIST dataset to use (default: OrganSMNIST)
export VISION_RAG_DATASET="PathMNIST"  # or ChestMNIST, DermaMNIST, etc.

# Choose CLIP model (default: clip-ViT-B-32)
export VISION_RAG_CLIP_MODEL="clip-ViT-L-14"

# Configure storage (defaults shown)
export VISION_RAG_COLLECTION_NAME="vision_rag"
export VISION_RAG_PERSIST_DIR="./chroma_db"
```

### Available Datasets

Vision RAG supports 12 MedMNIST datasets:

- **OrganSMNIST** (default): Organ images (sagittal view) - 11 classes
- **PathMNIST**: Colon pathology images - 9 classes
- **ChestMNIST**: Chest X-ray images - 14 classes
- **DermaMNIST**: Dermatoscope images of skin lesions - 7 classes
- **OCTMNIST**: Retinal OCT images - 4 classes
- **PneumoniaMNIST**: Chest X-ray for pneumonia detection - 2 classes
- **RetinaMNIST**: Fundus camera images for diabetic retinopathy - 5 classes
- **BreastMNIST**: Breast ultrasound images - 2 classes
- **BloodMNIST**: Blood cell microscope images - 8 classes
- **TissueMNIST**: Kidney cortex cells - 8 classes
- **OrganAMNIST**: Organ images (axial view) - 11 classes
- **OrganCMNIST**: Organ images (coronal view) - 11 classes

You can list available datasets programmatically:

```python
from vision_rag import list_available_datasets, get_dataset_config

# List all available datasets
datasets = list_available_datasets()
print(datasets)

# Get configuration for a specific dataset
config = get_dataset_config("PathMNIST")
print(f"Classes: {config['n_classes']}, Channels: {config['channels']}")
```

## Installation

This project uses UV for package management. Install dependencies with:

```bash
uv sync
```

### Activating the UV Environment

UV automatically manages virtual environments, but you can activate the environment in your shell for direct access to installed packages:

```bash
# Activate the UV environment in your current shell
source .venv/bin/activate

# Or use uv shell (if available in your uv version)
uv shell

# Run commands directly with uv (recommended approach)
uv run python your_script.py
uv run pytest
```

When the environment is activated, your shell prompt will typically show `(.venv)` at the beginning, indicating you're working within the virtual environment.

To deactivate the environment:

```bash
deactivate
```

**Note**: The recommended approach is to use `uv run` commands rather than activating the environment manually, as this ensures you're always using the correct environment and dependencies.

## Usage

### Basic Example with OrganSMNIST

```python
from vision_rag import (
    download_organmnist,
    load_organmnist_data,
    CLIPImageEncoder,
    ChromaRAGStore,
    ImageSearcher,
)

# Download dataset (uses permanent data directory by default)
download_organmnist()

# Load training data (automatically downloads if not present)
train_images, train_labels = load_organmnist_data(split="train")

# Initialize encoder
encoder = CLIPImageEncoder(model_name="clip-ViT-B-32")

# Encode images
embeddings = encoder.encode_images([img for img in train_images])

# Create RAG store
rag_store = ChromaRAGStore(
    collection_name="organmnist",
    persist_directory="./chroma_db",
)

# Add embeddings with metadata
metadatas = [{"label": int(label)} for label in train_labels]
rag_store.add_embeddings(embeddings, metadatas=metadatas)
```

### Using Different MedMNIST Datasets

```python
from vision_rag import (
    download_medmnist,
    load_medmnist_data,
    get_medmnist_label_names,
    CLIPImageEncoder,
    ChromaRAGStore,
    ImageSearcher,
)

# Download and load PathMNIST dataset
download_medmnist(dataset_name="PathMNIST")
train_images, train_labels = load_medmnist_data(dataset_name="PathMNIST", split="train")

# Get human-readable label names
label_names = get_medmnist_label_names(dataset_name="PathMNIST")
print(f"PathMNIST has {len(label_names)} classes: {label_names}")

# Initialize encoder
encoder = CLIPImageEncoder()

# Encode images
embeddings = encoder.encode_images([img for img in train_images])

# Create RAG store
rag_store = ChromaRAGStore(
    collection_name="pathmnist",
    persist_directory="./chroma_db_path",
)

# Add embeddings with metadata
metadatas = [{"label": int(label), "dataset": "PathMNIST"} for label in train_labels]
rag_store.add_embeddings(embeddings, metadatas=metadatas)

# Search for similar images
test_images, test_labels = load_medmnist_data(dataset_name="PathMNIST", split="test")
searcher = ImageSearcher(encoder=encoder, rag_store=rag_store)
results = searcher.search(test_images[0], n_results=5)

print(f"Found {len(results['ids'])} similar images")
print(f"IDs: {results['ids']}")
print(f"Distances: {results['distances']}")
```

### Using Environment Variables for Configuration

```python
import os
from vision_rag import (
    MEDMNIST_DATASET,
    CLIP_MODEL_NAME,
    download_medmnist,
    load_medmnist_data,
)

# Set environment variable before importing (or set in shell)
os.environ["VISION_RAG_DATASET"] = "ChestMNIST"
os.environ["VISION_RAG_CLIP_MODEL"] = "clip-ViT-L-14"

# Now the defaults will use ChestMNIST
print(f"Using dataset: {MEDMNIST_DATASET}")
print(f"Using CLIP model: {CLIP_MODEL_NAME}")

# Download and load using environment variable defaults
download_medmnist()  # Downloads ChestMNIST
train_images, train_labels = load_medmnist_data(split="train")  # Loads ChestMNIST

# Load test data
test_images, test_labels = load_organmnist_data(split="test", root="./data")

# Search for similar images
searcher = ImageSearcher(encoder=encoder, rag_store=rag_store)
results = searcher.search(test_images[0], n_results=5)

print(f"Found {len(results['ids'])} similar images")
print(f"IDs: {results['ids']}")
print(f"Distances: {results['distances']}")
```

## Testing

Run tests with pytest:

```bash
uv run pytest
```

Run specific test files:

```bash
uv run pytest tests/test_config.py
uv run pytest tests/test_data_loader.py
uv run pytest tests/test_encoder.py
uv run pytest tests/test_rag_store.py
uv run pytest tests/test_search.py
uv run pytest tests/test_integration.py
uv run pytest tests/test_visualization.py
uv run pytest tests/test_service.py
uv run pytest tests/test_mcp_server.py
```

### Code Coverage

Run tests with coverage report:

```bash
# Generate coverage report in terminal
uv run pytest tests/ --cov=vision_rag --cov-report=term-missing

# Generate HTML coverage report
uv run pytest tests/ --cov=vision_rag --cov-report=html

# Open HTML report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```


## Visualization

The system includes comprehensive visualization capabilities for analyzing the RAG pipeline:

```python
from vision_rag import RAGVisualizer

# Initialize visualizer
visualizer = RAGVisualizer(output_dir="./visualizations")

# Visualize sample input images going into RAG store
visualizer.save_input_images_grid(
    images=sample_images,
    labels=sample_labels,
    filename="input_to_rag.png"
)

# Visualize search query images
visualizer.save_search_input_images(
    images=query_images,
    labels=query_labels,
    filename="search_queries.png"
)

# Visualize search results
visualizer.save_search_results(
    query_image=query_img,
    query_label=query_label,
    retrieved_images=retrieved_images,
    retrieved_metadata=results['metadatas'],
    distances=results['distances'],
    filename="search_results.png"
)

# Visualize label distribution
visualizer.save_label_distribution(
    labels=all_labels,
    filename="label_distribution.png"
)

# Visualize embedding space using t-SNE or PCA
visualizer.save_embedding_space_visualization(
    embeddings=embeddings,
    labels=labels,
    method='tsne',
    filename="embedding_space.png"
)
```

### Demo Scripts

Run the comprehensive demonstration:

```bash
# Full demonstration with all visualizations
cd demonstrations && PYTHONPATH=.. uv run python demo_with_visualization.py

# Simple example
cd demonstrations && uv run python simple_visualization_example.py

# Multi-dataset demonstration (PathMNIST, ChestMNIST, etc.)
cd demonstrations && PYTHONPATH=.. uv run python multi_dataset_demo.py
```

## Project Structure

```
vision-rag/
├── vision_rag/
│   ├── __init__.py
│   ├── config.py            # Configuration and dataset definitions
│   ├── data_loader.py       # MedMNIST dataset loading (12+ datasets)
│   ├── encoder.py           # CLIP image encoder
│   ├── rag_store.py         # ChromaDB RAG store
│   ├── search.py            # Image search functionality
│   ├── visualization.py     # Visualization utilities
│   ├── service.py           # FastAPI REST service
│   └── mcp_server.py        # MCP agent server
├── demonstrations/
│   ├── demo_with_visualization.py    # Comprehensive demonstration
│   ├── simple_visualization_example.py  # Simple example
│   └── multi_dataset_demo.py         # Multi-dataset demonstration
├── examples/
│   └── client_demo.py       # Service client demo
├── tests/
│   ├── conftest.py
│   ├── test_basic.py
│   ├── test_config.py
│   ├── test_data_loader.py
│   ├── test_encoder.py
│   ├── test_rag_store.py
│   ├── test_search.py
│   ├── test_integration.py
│   ├── test_visualization.py
│   ├── test_service.py
│   └── test_mcp_server.py
├── run_service.py           # Service runner script
├── pyproject.toml
└── README.md
```

## Running as a Service

Vision RAG can be run as a service for agent-to-agent communication. The service uses the dataset configured via the `VISION_RAG_DATASET` environment variable.

### Service Configuration

```bash
# Configure which dataset the service should use
export VISION_RAG_DATASET="PathMNIST"  # Default: OrganSMNIST
export VISION_RAG_CLIP_MODEL="clip-ViT-B-32"
export VISION_RAG_COLLECTION_NAME="vision_rag_service"
export VISION_RAG_PERSIST_DIR="./service_chroma_db"
```

### 1. FastAPI REST Service

Start the REST API service:

```bash
# Basic startup (runs on port 8001 by default)
python run_service.py --mode api

# With specific dataset
VISION_RAG_DATASET="ChestMNIST" python run_service.py --mode api

# Custom host/port
python run_service.py --mode api --host 0.0.0.0 --port 8080

# Or use uvicorn directly
uv run uvicorn vision_rag.service:app --host 0.0.0.0 --port 8001
```

**API Endpoints:**
- `GET /` - Service information
- `GET /health` - Health check (includes dataset info)
- `GET /stats` - Get statistics
- `POST /search` - Search for similar images
- `POST /search/label` - Search by label (uses configured dataset's labels)
- `POST /add` - Add a single image
- `POST /add/batch` - Add multiple images
- `DELETE /clear` - Clear all embeddings

**API Documentation:**
Once running, visit `http://localhost:8001/docs` for interactive Swagger UI documentation.

### 2. MCP Agent Server

Start the Model Context Protocol server for agent communication:

```bash
python run_service.py --mode mcp

# With specific dataset
VISION_RAG_DATASET="DermaMNIST" python run_service.py --mode mcp
```

**Available MCP Tools:**
- `search_similar_images` - Search for visually similar medical images
- `search_by_label` - Search for images by label (dataset-specific)
- `add_image` - Add a medical image to the RAG store
- `get_statistics` - Get RAG store statistics
- `list_available_labels` - List all available labels for the configured dataset

### 3. Run Both Services

Start both FastAPI and MCP servers simultaneously:

```bash
python run_service.py --mode both --port 8001

# With specific dataset
VISION_RAG_DATASET="PathMNIST" python run_service.py --mode both --port 8001
```

### Example Client Usage

See `examples/client_demo.py` for a complete example of interacting with the service:

```python
import asyncio
import base64
import httpx

async def search_images():
    async with httpx.AsyncClient() as client:
        # Health check
        response = await client.get("http://localhost:8001/health")
        print(response.json())
        
        # Search for similar images
        response = await client.post(
            "http://localhost:8001/search",
            json={
                "image_base64": your_base64_image,
                "n_results": 5
            }
        )
        results = response.json()
        print(f"Found {results['count']} similar images")

asyncio.run(search_images())
```

### Agent Communication

Other AI agents can communicate with Vision RAG using the MCP protocol:

```python
from vision_rag.mcp_server import VisionRAGMCPServer

# Initialize MCP server
server = VisionRAGMCPServer()

# Call tools via MCP
result = await server.handle_tool_call(
    "search_similar_images",
    {"image_base64": "...", "n_results": 5}
)

# List available tools
tools = server.get_tool_definitions()
```



## Dependencies

- `medmnist`: MedMNIST dataset collection
- `sentence-transformers`: CLIP model implementation
- `chromadb`: Vector database for embeddings
- `fastapi`: Modern web framework for building APIs
- `uvicorn`: ASGI server for FastAPI
- `httpx`: Async HTTP client
- `pydantic`: Data validation and settings management
- `pytest`: Testing framework
- `pytest-cov`: Code coverage plugin
- `pillow`: Image processing
- `numpy`: Numerical operations
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical data visualization
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning utilities (t-SNE, PCA)
- `umap-learn`: UMAP dimensionality reduction

## Use Cases

### Medical Image Retrieval
- **Clinical Decision Support**: Find similar cases from historical data
- **Diagnostic Aid**: Retrieve examples of specific pathologies
- **Teaching & Training**: Build educational datasets

### AI Agent Integration
- **Multi-Agent Systems**: Vision RAG acts as a specialized image retrieval agent
- **Workflow Automation**: Agents can query medical images programmatically
- **Research Pipelines**: Automated image analysis and comparison

### Research Applications
- **Cohort Discovery**: Find patients with similar imaging patterns
- **Longitudinal Studies**: Track disease progression
- **Dataset Curation**: Build specialized image collections

