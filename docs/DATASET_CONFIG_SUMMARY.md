# Configurable MedMNIST Dataset Support

## Summary

Vision RAG now supports 12 different MedMNIST datasets through configurable environment variables and function parameters. This allows users to work with various medical imaging modalities beyond the original OrganSMNIST dataset.

## Changes Made

### 1. New Configuration Module (`vision_rag/config.py`)
- Added centralized configuration for all environment variables
- Defined 12 supported MedMNIST datasets with metadata
- Provides `get_dataset_config()` and `list_available_datasets()` functions

### 2. Enhanced Data Loader (`vision_rag/data_loader.py`)
- New generic functions: `download_medmnist()`, `load_medmnist_data()`, `get_medmnist_label_names()`
- All functions accept `dataset_name` parameter
- Updated `get_human_readable_label()` to accept dataset parameter

### 3. Updated Core Modules
- **encoder.py**: Uses `CLIP_MODEL_NAME` from config module
- **service.py**: Uses all config variables, passes dataset to label functions
- **mcp_server.py**: Uses config variables, supports dataset-specific labels
- **__init__.py**: Exports new configuration functions and constants

### 4. Comprehensive Tests
- New `test_config.py`: Tests configuration functionality
- Updated `test_data_loader.py`: Added tests for multi-dataset support
- All 55 tests passing

### 5. Documentation
- Updated README.md with:
  - Configuration section explaining environment variables
  - List of 12 supported datasets
  - Examples for using different datasets
  - Service configuration examples
- New demonstration script: `multi_dataset_demo.py`

## Supported Datasets

| Dataset | Description | Classes | Channels |
|---------|-------------|---------|----------|
| PathMNIST | Colon pathology images | 9 | 3 (RGB) |
| ChestMNIST | Chest X-ray images | 14 | 1 (Grayscale) |
| DermaMNIST | Skin lesion images | 7 | 3 (RGB) |
| OCTMNIST | Retinal OCT images | 4 | 1 (Grayscale) |
| PneumoniaMNIST | Pneumonia detection | 2 | 1 (Grayscale) |
| RetinaMNIST | Diabetic retinopathy | 5 | 3 (RGB) |
| BreastMNIST | Breast ultrasound | 2 | 1 (Grayscale) |
| BloodMNIST | Blood cells | 8 | 3 (RGB) |
| TissueMNIST | Kidney cortex cells | 8 | 1 (Grayscale) |
| OrganAMNIST | Organs (axial) | 11 | 1 (Grayscale) |
| OrganCMNIST | Organs (coronal) | 11 | 1 (Grayscale) |
| OrganSMNIST | Organs (sagittal) | 11 | 1 (Grayscale) |

## Usage Examples

### Environment Variable Configuration
```bash
export VISION_RAG_DATASET="PathMNIST"
export VISION_RAG_CLIP_MODEL="clip-ViT-L-14"
python your_script.py
```

### Programmatic Configuration
```python
from vision_rag import download_medmnist, load_medmnist_data

# Download and load PathMNIST
download_medmnist(dataset_name="PathMNIST")
images, labels = load_medmnist_data(dataset_name="PathMNIST", split="train")
```

### Service Configuration
```bash
# Run FastAPI service with ChestMNIST
VISION_RAG_DATASET="ChestMNIST" python scripts/run_service.py --mode api

# Run MCP server with DermaMNIST
VISION_RAG_DATASET="DermaMNIST" python scripts/run_service.py --mode mcp
```

## Testing

All tests pass (55 total):
```bash
uv run pytest tests/ -v
```

New tests added:
- `test_config.py`: 6 tests for configuration module
- `test_data_loader.py`: 4 additional tests for multi-dataset support

## Demonstration

Run the multi-dataset demo to see the new functionality in action:
```bash
cd demonstrations
PYTHONPATH=.. uv run python multi_dataset_demo.py
```

This demo:
1. Lists all available datasets
2. Downloads and loads PathMNIST
3. Builds a RAG store with PathMNIST data
4. Performs similarity search
5. Shows ChestMNIST configuration
