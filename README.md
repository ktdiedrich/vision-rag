# vision-rag

A retrieval-augmented generation (RAG) system for medical images using CLIP embeddings and ChromaDB.

## Overview

This project implements a vision-based RAG system that:
- Downloads the OrganSMNIST dataset from the MedMNIST collection
- Encodes medical images using the CLIP ViT-B-32 model
- Stores embeddings in ChromaDB for efficient retrieval
- Searches for similar images from the training set given test images

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

### Basic Example

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
uv run pytest tests/test_data_loader.py
uv run pytest tests/test_encoder.py
uv run pytest tests/test_rag_store.py
uv run pytest tests/test_search.py
uv run pytest tests/test_integration.py
uv run pytest tests/test_visualization.py
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
cd demonstrations && PYTHONPATH=../src uv run python demo_with_visualization.py

# Simple example
cd demonstrations && uv run python simple_visualization_example.py
```

## Project Structure

```
vision-rag/
├── src/
│   └── vision_rag/
│       ├── __init__.py
│       ├── data_loader.py       # OrganSMNIST dataset loading
│       ├── encoder.py           # CLIP image encoder
│       ├── rag_store.py         # ChromaDB RAG store
│       ├── search.py            # Image search functionality
│       └── visualization.py     # Visualization utilities
├── demonstrations/
│   ├── demo_with_visualization.py    # Comprehensive demonstration
│   └── simple_visualization_example.py  # Simple example
├── tests/
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_encoder.py
│   ├── test_rag_store.py
│   ├── test_search.py
│   ├── test_integration.py
│   └── test_visualization.py
├── pyproject.toml
└── README.md
```

## Dependencies

- `medmnist`: MedMNIST dataset collection
- `sentence-transformers`: CLIP model implementation
- `chromadb`: Vector database for embeddings
- `pytest`: Testing framework
- `pillow`: Image processing
- `numpy`: Numerical operations
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical data visualization
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning utilities (t-SNE, PCA)