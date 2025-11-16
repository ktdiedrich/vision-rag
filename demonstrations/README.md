# Vision RAG Demonstrations

This directory contains demonstration scripts that showcase the vision RAG system capabilities with comprehensive visualizations.

## Scripts

### 1. `simple_visualization_demo.py`
A quick demonstration that shows the core functionality:

- Uses 50 training images for fast execution
- Creates 6 visualization files including t-SNE embedding plot
- Perfect for understanding basic concepts

**Run:**
```bash
uv run python simple_visualization_demo.py
```

**Output:** Creates `output/simple_visualizations/` directory with:
- `input_to_rag_store.png` - Sample training images
- `embedding_space_tsne.png` - t-SNE visualization of embeddings
- `search_query_images.png` - Query images for search
- `search_results_1-5.png` - Search results with similarities

### 2. `demo_with_visualization.py`
Comprehensive demonstration with full analysis:

- Uses 1000 training images for robust results
- Creates 9 visualization files
- Includes embedding space analysis
- Shows multiple search scenarios

**Run:**
```bash
PYTHONPATH=.. uv run python demo_with_visualization.py
```

**Output:** Creates `output/visualizations/` directory with:
- `01_input_images_to_rag.png` - Sample training images grid
- `02_label_distribution.png` - Dataset label distribution
- `03_embedding_space_tsne.png` - t-SNE visualization of embeddings
- `04_search_input_images.png` - Query images grid
- `05_search_results_1-5.png` - Multiple search result comparisons

## Generated Files

Both scripts create their own output directories to avoid conflicts:
- Simple example → `output/simple_visualizations/`
- Full demo → `output/visualizations/`

## Requirements

Make sure you have all dependencies installed:
```bash
cd .. && uv sync
```

The scripts will automatically:
1. Download the OrganSMNIST dataset (if not present)
2. Create necessary directories
3. Generate all visualization files
4. Print progress and file locations

## Understanding the Visualizations

### Input Images (`input_to_rag_store.png`)
Shows sample images that are encoded and stored in the RAG system, with their corresponding labels.

### Search Queries (`search_query_images.png`) 
Shows the test images used as queries to search the RAG store.

### Search Results (`search_results_X.png`)
Shows each query image (highlighted in red) alongside the top 5 most similar images retrieved from the RAG store, with similarity distances.

### Label Distribution (`label_distribution.png`)
Bar chart showing how many images of each class are in the training set.

### Embedding Space (`embedding_space_tsne.png`)
2D projection of the high-dimensional CLIP embeddings, colored by class labels, showing how similar images cluster together.