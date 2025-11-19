# Dataset Preloading Guide

This guide explains how to use the dataset preloading feature to quickly populate your Vision RAG store with MedMNIST datasets.

## Overview

The preload functionality allows you to:
- Discover available MedMNIST datasets
- Automatically download datasets (if not already cached)
 - Encode images using the configured image encoder (CLIP or DINO)
- Store embeddings in ChromaDB
- Load multiple datasets simultaneously
- Control dataset size and number of images

## Available Endpoints

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/datasets` | GET | List all available MedMNIST datasets |
| `/preload` | POST | Preload a dataset into the RAG store |
| `/stats` | GET | Get current store statistics |
| `/labels` | GET | Get available labels for current dataset |
| `/clear` | DELETE | Clear all embeddings from store |

### MCP Tools

| Tool | Description |
|------|-------------|
| `list_available_datasets` | List all available MedMNIST datasets |
| `preload_dataset` | Preload a dataset into the RAG store |
| `get_statistics` | Get RAG store statistics |
| `list_available_labels` | List available labels |

## Usage Examples

### Using the REST API

#### 1. List Available Datasets

```bash
curl http://localhost:8001/datasets
```

```json
{
  "datasets": {
    "PathMNIST": {
      "description": "Colon pathology images",
      "n_classes": 9,
      "image_size": 224,
      "channels": 3
    },
    "ChestMNIST": {
      "description": "Chest X-ray images",
      "n_classes": 14,
      "image_size": 224,
      "channels": 1
    },
    ...
  },
  "count": 12
}
```

#### 2. Preload a Dataset

**Load entire training set:**
```bash
curl -X POST http://localhost:8001/preload \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "PathMNIST",
    "split": "train"
  }'
```

**Load limited number of images:**
```bash
curl -X POST http://localhost:8001/preload \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "PneumoniaMNIST",
    "split": "train",
    "max_images": 100
  }'
```

**Specify dataset size:**
```bash
curl -X POST http://localhost:8001/preload \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "BreastMNIST",
    "split": "test",
    "size": 28,
    "max_images": 50
  }'
```

**Response:**
```json
{
  "status": "success",
  "dataset_name": "PathMNIST",
  "split": "train",
  "images_loaded": 89996,
  "total_embeddings": 89996,
  "message": "Successfully loaded 89996 images from PathMNIST (train)"
}
```

#### 3. Check Statistics

```bash
curl http://localhost:8001/stats
```

```json
{
  "total_embeddings": 89996,
  "total_images": 89996,
  "collection_name": "vision_rag",
  "persist_directory": "./chroma_db_api",
  "image_store_directory": "./image_store_api"
}
```

### Using Python with the API

```python
import requests

API_URL = "http://localhost:8001"

# List available datasets
response = requests.get(f"{API_URL}/datasets")
datasets = response.json()
print(f"Available datasets: {datasets['count']}")

# Preload PneumoniaMNIST
response = requests.post(
    f"{API_URL}/preload",
    json={
        "dataset_name": "PneumoniaMNIST",
        "split": "train",
        "max_images": 100
    }
)
result = response.json()
print(f"Loaded {result['images_loaded']} images")

# Get statistics
response = requests.get(f"{API_URL}/stats")
stats = response.json()
print(f"Total embeddings: {stats['total_embeddings']}")
```

### Using the MCP Server

```python
import asyncio
from vision_rag.mcp_server import VisionRAGMCPServer

async def preload_example():
    server = VisionRAGMCPServer()
    
    # List available datasets
    datasets = await server.list_available_datasets()
    print(f"Available: {datasets['count']} datasets")
    
    # Preload a dataset
    result = await server.preload_dataset(
        dataset_name="PathMNIST",
        split="train",
        max_images=1000  # Limit for demo
    )
    
    if result.get("success"):
        print(f"Loaded {result['images_loaded']} images")
        print(f"Total in store: {result['total_embeddings']}")
    else:
        print(f"Error: {result.get('error')}")

asyncio.run(preload_example())
```

## Available Datasets

| Dataset | Description | Classes | Channels |
|---------|-------------|---------|----------|
| **PathMNIST** | Colon pathology images | 9 | RGB (3) |
| **ChestMNIST** | Chest X-ray images | 14 | Grayscale (1) |
| **DermaMNIST** | Skin lesion images | 7 | RGB (3) |
| **OCTMNIST** | Retinal OCT images | 4 | Grayscale (1) |
| **PneumoniaMNIST** | Pneumonia detection | 2 | Grayscale (1) |
| **RetinaMNIST** | Diabetic retinopathy | 5 | RGB (3) |
| **BreastMNIST** | Breast ultrasound | 2 | Grayscale (1) |
| **BloodMNIST** | Blood cell images | 8 | RGB (3) |
| **TissueMNIST** | Kidney cortex cells | 8 | Grayscale (1) |
| **OrganAMNIST** | Organs (axial) | 11 | Grayscale (1) |
| **OrganCMNIST** | Organs (coronal) | 11 | Grayscale (1) |
| **OrganSMNIST** | Organs (sagittal) | 11 | Grayscale (1) |

## Dataset Splits

Each MedMNIST dataset has three splits:
- **train**: Training data (largest, typically ~90% of data)
- **test**: Test data (typically ~10% of data)
- **val**: Validation data (small validation set)

## Image Sizes

MedMNIST datasets are available in multiple resolutions:
- **28×28**: Original size (fastest)
- **64×64**: Medium resolution
- **128×128**: High resolution
- **224×224**: Highest resolution (default, matches CLIP)

**Recommendation:** Use 224×224 for best quality with CLIP or DINO embeddings — this size matches many transformer encoders' expected input and is the default for MedMNIST.

## Performance Considerations

### Loading Speed

| Size | Images/sec (approx) | 10k images |
|------|-------------------|-----------|
| 28×28 | ~500 | 20 seconds |
| 64×64 | ~300 | 33 seconds |
| 128×128 | ~150 | 67 seconds |
| 224×224 | ~100 | 100 seconds |

*Performance varies based on CPU, GPU availability, and disk speed.*

### Memory Usage

- **CLIP Model**: ~350 MB
- **ChromaDB**: ~100 MB base + ~512 bytes per embedding
- **Images**: Depends on size and compression

**Example:** 10,000 images at 224×224:
- Embeddings: ~5 MB (512 dimensions × 4 bytes × 10k)
- Images on disk: ~50-100 MB (PNG compressed)
- Total RAM during encoding: ~2-3 GB peak

### Disk Usage

Preloading stores:
1. **Original dataset files** (in `data/` directory)
2. **Saved images** (in `image_store_*/` directory)
3. **Embeddings** (in `chroma_db_*/` directory)

**Example for PathMNIST (90k training images):**
- Dataset file: ~400 MB
- Saved images: ~900 MB
- Embeddings: ~45 MB
- **Total**: ~1.35 GB

## Best Practices

### 1. Start Small
```python
# Test with a small subset first
await server.preload_dataset(
    dataset_name="PneumoniaMNIST",
    split="train",
    max_images=100  # Start small
)
```

### 2. Use Appropriate Size
```python
# Use 224×224 for production (best quality)
# Use 28×28 for development (fastest)
await server.preload_dataset(
    dataset_name="PathMNIST",
    split="train",
    size=28  # Fast for testing
)
```

### 3. Load Multiple Datasets
```python
datasets = [
    ("PneumoniaMNIST", "train", 500),
    ("BreastMNIST", "train", 500),
    ("BloodMNIST", "test", 200),
]

for name, split, max_imgs in datasets:
    result = await server.preload_dataset(
        dataset_name=name,
        split=split,
        max_images=max_imgs
    )
    print(f"Loaded {result['images_loaded']} from {name}")
```

### 4. Monitor Progress
```python
# Get stats before and after
stats_before = await server.get_statistics()
print(f"Before: {stats_before['total_embeddings']} embeddings")

await server.preload_dataset("PathMNIST", "train")

stats_after = await server.get_statistics()
print(f"After: {stats_after['total_embeddings']} embeddings")
print(f"Added: {stats_after['total_embeddings'] - stats_before['total_embeddings']}")
```

### 5. Clear When Needed
```python
# Clear the store to start fresh
response = requests.delete("http://localhost:8001/clear")
print(response.json())  # {"status": "success", "message": "All embeddings cleared"}
```

## Example Workflows

### Research Workflow
```python
# 1. Clear existing data
requests.delete("http://localhost:8001/clear")

# 2. Load specific dataset for research
requests.post("http://localhost:8001/preload", json={
    "dataset_name": "ChestMNIST",
    "split": "train",
    "size": 224
})

# 3. Run experiments
# ... your research code ...
```

### Multi-Dataset Application
```python
# Load multiple datasets for comprehensive medical image search
datasets = ["PathMNIST", "ChestMNIST", "DermaMNIST", "RetinaMNIST"]

for dataset in datasets:
    requests.post("http://localhost:8001/preload", json={
        "dataset_name": dataset,
        "split": "train",
        "max_images": 1000  # Balanced across datasets
    })
```

### Quick Demo Setup
```python
# Load small subsets of diverse datasets for demonstration
demo_datasets = {
    "PneumoniaMNIST": 50,
    "BreastMNIST": 50,
    "BloodMNIST": 50,
}

for dataset, count in demo_datasets.items():
    requests.post("http://localhost:8001/preload", json={
        "dataset_name": dataset,
        "split": "train",
        "max_images": count,
        "size": 28  # Fast loading
    })
```

## Troubleshooting

### Dataset Not Found
```
Error: Dataset file not found
```
**Solution:** The dataset will be automatically downloaded on first load. Ensure you have internet connection and sufficient disk space.

### Out of Memory
```
Error: Encoding failed - out of memory
```
**Solution:** Reduce `max_images` or use smaller image size (28 or 64).

### Slow Loading
**Solution:** 
- Use smaller image size for testing (28 or 64)
- Load fewer images with `max_images`
- Ensure you have adequate disk space and RAM

### Dataset Already Loaded
Multiple preload calls will add images to the store. To start fresh:
```python
# Clear before preloading
requests.delete("http://localhost:8001/clear")
requests.post("http://localhost:8001/preload", json={"dataset_name": "PathMNIST", "split": "train"})
```

## Demo Scripts

Try the included demo scripts:

**MCP Server Demo:**
```bash
python examples/preload_example.py
```

**REST API Demo:**
```bash
# Terminal 1: Start the API
python -m vision_rag.service

# Terminal 2: Run the demo
python examples/api_preload_example.py
```

## See Also

- [Dataset Configuration Guide](DATASET_CONFIG_SUMMARY.md)
- [Service Deployment Guide](SERVICE_GUIDE.md)
- [API Documentation](../README.md#api-endpoints)
