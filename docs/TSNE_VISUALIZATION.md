# t-SNE Visualization for RAG Store

This document describes how to generate t-SNE (and other dimensionality reduction) visualizations of embeddings stored in the Vision RAG system's ChromaDB store.

## Overview

The Vision RAG system now supports generating 2D visualizations of the high-dimensional CLIP embeddings stored in the RAG database. This helps you:

- **Understand clustering**: See how similar images group together in embedding space
- **Identify patterns**: Discover semantic relationships between images
- **Debug embeddings**: Verify that similar images have similar embeddings
- **Explore datasets**: Visualize the distribution of different classes/labels

## Available Methods

### 1. MCP Server Tool

The MCP server includes a `generate_tsne_plot` tool that can be called by other agents.

#### Tool Definition

```python
{
    "name": "generate_tsne_plot",
    "description": "Generate a t-SNE (or PCA/UMAP) visualization of all embeddings",
    "parameters": {
        "output_filename": "tsne_visualization.png",  # Output file path
        "method": "tsne",  # 'tsne', 'pca', or 'umap'
        "title": "RAG Store Embedding Space Visualization"  # Plot title
    }
}
```

#### Usage Example

```python
from vision_rag.mcp_server import VisionRAGMCPServer

# Initialize server
server = VisionRAGMCPServer()

# Generate t-SNE plot
result = await server.generate_tsne_plot(
    output_filename="my_tsne_plot.png",
    method="tsne",
    title="My Embedding Visualization"
)

print(result)
# {
#     "success": True,
#     "output_path": "./my_tsne_plot.png",
#     "total_embeddings": 1000,
#     "method": "tsne",
#     "unique_labels": 11,
#     "message": "Successfully generated TSNE plot with 1000 embeddings"
# }
```

### 2. FastAPI Endpoint

The FastAPI service exposes a REST endpoint for generating visualizations.

#### Endpoint: `POST /visualize/tsne`

**Request Body:**
```json
{
    "output_filename": "tsne_visualization.png",
    "method": "tsne",
    "title": "RAG Store Embedding Space Visualization"
}
```

**Response:**
```json
{
    "success": true,
    "output_path": "./tsne_visualization.png",
    "total_embeddings": 1000,
    "method": "tsne",
    "unique_labels": 11,
    "message": "Successfully generated TSNE plot with 1000 embeddings"
}
```

#### cURL Examples

```bash
# Generate t-SNE plot
curl -X POST http://localhost:8001/visualize/tsne \
  -H "Content-Type: application/json" \
  -d '{
    "output_filename": "tsne.png",
    "method": "tsne",
    "title": "t-SNE Visualization"
  }'

# Generate PCA plot
curl -X POST http://localhost:8001/visualize/tsne \
  -H "Content-Type: application/json" \
  -d '{
    "output_filename": "pca.png",
    "method": "pca"
  }'

# Generate UMAP plot
curl -X POST http://localhost:8001/visualize/tsne \
  -H "Content-Type: application/json" \
  -d '{
    "output_filename": "umap.png",
    "method": "umap"
  }'
```

#### Python Requests Example

```python
import requests

response = requests.post(
    "http://localhost:8001/visualize/tsne",
    json={
        "output_filename": "my_visualization.png",
        "method": "tsne",
        "title": "My Custom Title"
    }
)

result = response.json()
if result["success"]:
    print(f"Plot saved to: {result['output_path']}")
else:
    print(f"Error: {result['error']}")
```

## Dimensionality Reduction Methods

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Best for**: Visualizing local structure and clusters

**Characteristics**:
- Preserves local neighborhood relationships
- Good at showing clusters
- Can be slow for large datasets
- Non-deterministic (results vary between runs)

**When to use**: When you want to see how images cluster together and identify distinct groups.

### PCA (Principal Component Analysis)

**Best for**: Fast visualization of global structure

**Characteristics**:
- Linear transformation
- Very fast
- Preserves global structure
- Deterministic (same results every time)

**When to use**: When you need a quick overview or have very large datasets.

### UMAP (Uniform Manifold Approximation and Projection)

**Best for**: Balanced view of local and global structure

**Characteristics**:
- Faster than t-SNE
- Preserves both local and global structure
- More stable than t-SNE
- Requires `umap-learn` package

**When to use**: When you want both cluster detail and overall structure.

**Note**: UMAP requires installing the `umap-learn` package. If not available, the system will fall back to PCA.

## Output Format

All methods generate:
- **PNG image** with 150 DPI resolution
- **2D scatter plot** with points colored by label
- **Colorbar** showing label mappings (if ≤11 unique labels)
- **Grid and axes** for easy interpretation

## Examples

### Complete Workflow

```python
from vision_rag.mcp_server import VisionRAGMCPServer
from vision_rag.data_loader import load_medmnist_data, get_image_from_array

# 1. Initialize server
server = VisionRAGMCPServer()

# 2. Load and add images (if needed)
if server.rag_store.count() == 0:
    images, labels = load_medmnist_data("OrganSMNIST", split="train")
    pil_images = [get_image_from_array(img) for img in images[:100]]
    
    embeddings = server.encoder.encode_images(pil_images)
    metadatas = [{"label": int(labels[i])} for i in range(100)]
    
    server.rag_store.add_embeddings(embeddings, metadatas=metadatas)

# 3. Generate visualizations
await server.generate_tsne_plot(
    output_filename="tsne_100_images.png",
    method="tsne"
)

await server.generate_tsne_plot(
    output_filename="pca_100_images.png",
    method="pca"
)
```

### Using with Different Datasets

```python
# Generate separate visualizations for different datasets
datasets = ["PathMNIST", "ChestMNIST", "OrganSMNIST"]

for dataset in datasets:
    # Preload dataset
    await server.preload_dataset(
        dataset_name=dataset,
        split="train",
        max_images=200
    )
    
    # Generate visualization
    await server.generate_tsne_plot(
        output_filename=f"{dataset.lower()}_tsne.png",
        title=f"{dataset} Embedding Space"
    )
    
    # Clear for next dataset
    await server.clear_store(clear_embeddings=True)
```

## Troubleshooting

### No embeddings found

**Error**: `"No embeddings found in RAG store"`

**Solution**: Add images to the RAG store first using:
- `preload_dataset` tool/endpoint
- `add_image` tool/endpoint
- Direct embedding addition

### Plot looks cluttered

**Problem**: Too many unique labels or overlapping points

**Solutions**:
- Use a subset of data (`max_images` parameter)
- Try different methods (PCA for overview, t-SNE for detail)
- Adjust figure size in the visualization module

### UMAP not available

**Error**: Falls back to PCA when UMAP requested

**Solution**: Install UMAP:
```bash
uv pip install umap-learn
```

### Slow generation

**Problem**: t-SNE takes a long time with many embeddings

**Solutions**:
- Use PCA instead (much faster)
- Reduce dataset size
- Use UMAP (faster than t-SNE)

## Implementation Details

### Added Methods

**RAG Store** (`rag_store.py`):
- `get_all_embeddings()`: Retrieves all embeddings and metadata

**MCP Server** (`mcp_server.py`):
- `generate_tsne_plot()`: Async tool for generating visualizations

**FastAPI Service** (`service.py`):
- `POST /visualize/tsne`: REST endpoint for visualizations

### Dependencies

The feature uses existing visualization infrastructure:
- `RAGVisualizer` class for plot generation
- `sklearn.manifold.TSNE` for t-SNE
- `sklearn.decomposition.PCA` for PCA
- `umap-learn` (optional) for UMAP

## See Also

- [RAG Visualization Module](../vision_rag/visualization.py)
- [MCP Server Documentation](API.md)
- [FastAPI Service Documentation](API.md)
- [Demo Script](../demonstrations/tsne_visualization_demo.py)
