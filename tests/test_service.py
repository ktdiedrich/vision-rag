"""Tests for FastAPI service module."""

import pytest
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from vision_rag.service import app
from vision_rag.utils import encode_image_to_base64


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple 28x28 grayscale image
    img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
    return Image.fromarray(img_array, mode='L')


@pytest.fixture
def sample_image_base64(sample_image):
    """Create a base64 encoded sample image."""
    return encode_image_to_base64(sample_image)


@pytest.fixture
def setup_test_data(client):
    """Add some test data to the service."""
    # Create and add test images one by one
    for i in range(5):
        img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        
        add_request = {
            "image_base64": encode_image_to_base64(img),
            "metadata": {"label": i % 3, "test": True}
        }
        
        response = client.post("/add", json=add_request)
        assert response.status_code == 200
    
    yield
    
    # Cleanup after test
    client.delete("/clear")


class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "description" in data
        assert data["service"] == "Vision RAG API"


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "encoder_model" in data
        assert "collection_name" in data
        assert "embeddings_count" in data
        assert isinstance(data["embeddings_count"], int)


class TestStatsEndpoint:
    """Tests for the statistics endpoint."""
    
    def test_get_stats(self, client):
        """Test getting service statistics."""
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_embeddings" in data
        assert "collection_name" in data
        assert "persist_directory" in data
        assert isinstance(data["total_embeddings"], int)
    
    def test_stats_after_adding_data(self, client, sample_image_base64):
        """Test stats change after adding data."""
        # Get initial count
        response = client.get("/stats")
        initial_count = response.json()["total_embeddings"]
        
        # Add an image
        add_request = {
            "image_base64": sample_image_base64,
            "metadata": {"label": 0}
        }
        client.post("/add", json=add_request)
        
        # Check count increased
        response = client.get("/stats")
        new_count = response.json()["total_embeddings"]
        assert new_count == initial_count + 1


class TestLabelsEndpoint:
    """Tests for the labels endpoint."""
    
    def test_get_labels_success(self, client):
        """Test getting labels when dataset is available."""
        response = client.get("/labels")
        
        # Should succeed if dataset is available (200) or fail with 404 if not downloaded
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "labels" in data
            assert "count" in data
            assert "dataset" in data
            assert isinstance(data["labels"], dict)
            assert isinstance(data["count"], int)
            assert data["count"] == len(data["labels"])
    
    def test_get_labels_file_not_found(self, client, monkeypatch):
        """Test labels endpoint returns 404 when dataset file doesn't exist."""
        from vision_rag import service
        
        def mock_get_medmnist_label_names(*args, **kwargs):
            raise FileNotFoundError(
                "OrganSMNIST dataset (size=224) not found at /path/to/data/organsmnist_224.npz. "
                "Please download the dataset first by calling download_medmnist('OrganSMNIST', size=224) "
                "or load_medmnist_data('OrganSMNIST', size=224)."
            )
        
        monkeypatch.setattr(service, "get_medmnist_label_names", mock_get_medmnist_label_names)
        
        response = client.get("/labels")
        assert response.status_code == 404
        
        data = response.json()
        assert "detail" in data
        assert "Dataset not found" in data["detail"]
        assert "OrganSMNIST" in data["detail"]
    
    def test_get_labels_other_error(self, client, monkeypatch):
        """Test labels endpoint returns 500 for other errors."""
        from vision_rag import service
        
        def mock_get_medmnist_label_names(*args, **kwargs):
            raise ValueError("Invalid dataset configuration")
        
        monkeypatch.setattr(service, "get_medmnist_label_names", mock_get_medmnist_label_names)
        
        response = client.get("/labels")
        assert response.status_code == 500
        
        data = response.json()
        assert "detail" in data
        assert "Failed to retrieve labels" in data["detail"]


class TestSearchEndpoint:
    """Tests for the search endpoint."""
    
    def test_search_without_data(self, client, sample_image_base64):
        """Test search with empty database."""
        # Clear database first
        client.delete("/clear")
        
        search_request = {
            "image_base64": sample_image_base64,
            "n_results": 5
        }
        
        response = client.post("/search", json=search_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["count"] == 0
        assert len(data["results"]) == 0
    
    def test_search_with_data(self, client, sample_image_base64, setup_test_data):
        """Test search with data in database."""
        search_request = {
            "image_base64": sample_image_base64,
            "n_results": 3
        }
        
        response = client.post("/search", json=search_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["count"] <= 3  # May be less if fewer images in DB
        assert "query_info" in data
        assert "results" in data
        
        # Check query info
        assert "image_size" in data["query_info"]
        assert "n_results_requested" in data["query_info"]
        
        # Check results format
        if data["count"] > 0:
            result = data["results"][0]
            assert "id" in result
            assert "distance" in result
            assert "metadata" in result
    
    def test_search_invalid_base64(self, client):
        """Test search with invalid base64 data."""
        search_request = {
            "image_base64": "invalid_base64_string",
            "n_results": 5
        }
        
        response = client.post("/search", json=search_request)
        assert response.status_code == 400
    
    def test_search_n_results_validation(self, client, sample_image_base64):
        """Test n_results parameter validation."""
        # Test n_results too small
        search_request = {
            "image_base64": sample_image_base64,
            "n_results": 0
        }
        response = client.post("/search", json=search_request)
        assert response.status_code == 422  # Validation error
        
        # Test n_results too large
        search_request["n_results"] = 101
        response = client.post("/search", json=search_request)
        assert response.status_code == 422


class TestSearchByLabelEndpoint:
    """Tests for the search by label endpoint."""
    
    def test_search_by_label(self, client, setup_test_data):
        """Test searching by label."""
        search_request = {
            "label": 0,
            "n_results": 10
        }
        
        response = client.post("/search/label", json=search_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert "count" in data
        assert "query_info" in data
        
        # All results should have the requested label
        for result in data["results"]:
            assert result["metadata"]["label"] == 0
    
    def test_search_by_label_no_matches(self, client, setup_test_data):
        """Test search by label with no matches."""
        search_request = {
            "label": 999,  # Non-existent label
            "n_results": 10
        }
        
        response = client.post("/search/label", json=search_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["count"] == 0
        assert len(data["results"]) == 0
    
    def test_search_by_label_with_limit(self, client, setup_test_data):
        """Test search by label with result limit."""
        search_request = {
            "label": 0,
            "n_results": 1
        }
        
        response = client.post("/search/label", json=search_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["count"] <= 1


class TestAddImageEndpoint:
    """Tests for the add image endpoint."""
    
    def test_add_single_image(self, client, sample_image_base64):
        """Test adding a single image."""
        add_request = {
            "image_base64": sample_image_base64,
            "metadata": {"label": 5, "source": "test"}
        }
        
        response = client.post("/add", json=add_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert "status" in data
        assert data["status"] == "success"
    
    def test_add_image_without_metadata(self, client, sample_image_base64):
        """Test adding image without metadata."""
        add_request = {
            "image_base64": sample_image_base64
        }
        
        response = client.post("/add", json=add_request)
        assert response.status_code == 200
    
    def test_add_image_invalid_base64(self, client):
        """Test adding image with invalid base64."""
        add_request = {
            "image_base64": "not_valid_base64",
            "metadata": {"label": 0}
        }
        
        response = client.post("/add", json=add_request)
        assert response.status_code == 400


class TestAddBatchEndpoint:
    """Tests for the batch add endpoint."""
    
    def test_add_batch_images(self, client):
        """Test adding multiple images in batch."""
        # Create multiple test images as files
        files = []
        for i in range(3):
            img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L')
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            
            files.append(("files", ("test_image.png", buffer, "image/png")))
        
        response = client.post("/add/batch", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert "count" in data
        assert data["count"] == 3
        assert "status" in data
    
    def test_add_batch_without_files(self, client):
        """Test batch add without any files."""
        response = client.post("/add/batch", files=[])
        assert response.status_code == 422  # Validation error - missing required files


class TestClearEndpoint:
    """Tests for the clear endpoint."""
    
    def test_clear_embeddings(self, client, sample_image_base64):
        """Test clearing all embeddings."""
        # Add some data first
        add_request = {
            "image_base64": sample_image_base64,
            "metadata": {"label": 0}
        }
        client.post("/add", json=add_request)
        
        # Verify data exists
        stats = client.get("/stats").json()
        initial_count = stats["total_embeddings"]
        assert initial_count > 0
        
        # Clear all data
        response = client.delete("/clear")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "cleared" in data["message"].lower()
        
        # Verify data is cleared
        stats = client.get("/stats").json()
        assert stats["total_embeddings"] == 0


class TestEndToEndWorkflow:
    """Test complete workflow of adding and searching images."""
    
    def test_complete_workflow(self, client):
        """Test adding images and searching for similar ones."""
        # 1. Clear any existing data
        client.delete("/clear")
        
        # 2. Add test images using individual add endpoint
        images = []
        for i in range(5):
            img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L')
            images.append(img)
            
            add_request = {
                "image_base64": encode_image_to_base64(img),
                "metadata": {"label": i % 2, "index": i}
            }
            
            add_response = client.post("/add", json=add_request)
            assert add_response.status_code == 200
        
        # 3. Verify stats
        stats = client.get("/stats").json()
        assert stats["total_embeddings"] == 5
        
        # 4. Search for similar images
        search_request = {
            "image_base64": encode_image_to_base64(images[0]),
            "n_results": 3
        }
        
        search_response = client.post("/search", json=search_request)
        assert search_response.status_code == 200
        
        results = search_response.json()
        assert results["count"] <= 3
        assert len(results["results"]) <= 3
        
        # 5. Search by label
        label_search = {
            "label": 0,
            "n_results": 10
        }
        
        label_response = client.post("/search/label", json=label_search)
        assert label_response.status_code == 200
        
        label_results = label_response.json()
        assert label_results["count"] == 3  # Labels 0, 2, 4
        
        # 6. Clean up
        client.delete("/clear")
        final_stats = client.get("/stats").json()
        assert final_stats["total_embeddings"] == 0


class TestErrorHandling:
    """Test error handling in various scenarios."""
    
    def test_malformed_json(self, client):
        """Test handling of malformed JSON requests."""
        response = client.post(
            "/search",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        # Missing image_base64 field
        response = client.post("/search", json={"n_results": 5})
        assert response.status_code == 422
        
        # Missing label field
        response = client.post("/search/label", json={"n_results": 5})
        assert response.status_code == 422


class TestConcurrency:
    """Test concurrent operations to ensure thread safety."""
    
    def test_concurrent_image_additions(self, client):
        """Test that concurrent image additions produce unique IDs."""
        import concurrent.futures
        
        # Clear database first
        client.delete("/clear")
        
        def add_image_task(index):
            """Task to add an image."""
            img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L')
            
            add_request = {
                "image_base64": encode_image_to_base64(img),
                "metadata": {"task_index": index}
            }
            
            response = client.post("/add", json=add_request)
            return response.json()
        
        # Add 10 images concurrently
        num_images = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(add_image_task, i) for i in range(num_images)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # Verify all additions were successful
        assert len(results) == num_images
        for result in results:
            assert result["status"] == "success"
            assert "id" in result
        
        # Verify all IDs are unique
        ids = [result["id"] for result in results]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"
        
        # Verify final count matches number of images added
        stats = client.get("/stats").json()
        assert stats["total_embeddings"] == num_images
        
        # Clean up
        client.delete("/clear")


class TestPreloadEndpoint:
    """Comprehensive tests for the preload dataset endpoint."""
    
    def test_preload_dataset_success(self, client):
        """Test successful dataset preloading."""
        # Clear first
        client.delete("/clear")
        
        request = {
            "dataset_name": "PneumoniaMNIST",
            "split": "train",
            "max_images": 5,
            "size": 28
        }
        
        response = client.post("/preload", json=request)
        
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            assert data["dataset_name"] == "PneumoniaMNIST"
            assert data["split"] == "train"
            assert data["images_loaded"] <= 5
            assert data["total_embeddings"] >= data["images_loaded"]
    
    def test_preload_invalid_dataset(self, client):
        """Test preloading with invalid dataset name."""
        request = {
            "dataset_name": "InvalidDatasetName",
            "split": "train",
            "max_images": 5
        }
        
        response = client.post("/preload", json=request)
        assert response.status_code == 400
        assert "Unknown dataset" in response.json()["detail"]
    
    def test_preload_invalid_split(self, client):
        """Test preloading with invalid split."""
        request = {
            "dataset_name": "PneumoniaMNIST",
            "split": "invalid_split",
            "max_images": 5
        }
        
        response = client.post("/preload", json=request)
        assert response.status_code == 400
        assert "Invalid split" in response.json()["detail"]
    
    def test_preload_multiple_datasets(self, client):
        """Test preloading multiple datasets sequentially."""
        client.delete("/clear")
        
        # Preload first dataset
        request1 = {
            "dataset_name": "PneumoniaMNIST",
            "split": "train",
            "max_images": 3,
            "size": 28
        }
        
        response1 = client.post("/preload", json=request1)
        
        if response1.status_code == 200:
            count_after_first = response1.json()["total_embeddings"]
            
            # Preload second dataset
            request2 = {
                "dataset_name": "BreastMNIST",
                "split": "train",
                "max_images": 3,
                "size": 28
            }
            
            response2 = client.post("/preload", json=request2)
            
            if response2.status_code == 200:
                count_after_second = response2.json()["total_embeddings"]
                assert count_after_second >= count_after_first
    
    def test_preload_with_different_sizes(self, client):
        """Test preloading with different image sizes."""
        for size in [28, 64]:
            request = {
                "dataset_name": "PneumoniaMNIST",
                "split": "train",
                "max_images": 2,
                "size": size
            }
            
            response = client.post("/preload", json=request)
            if response.status_code == 200:
                data = response.json()
                assert data["status"] == "success"


class TestClearEndpoint:
    """Comprehensive tests for the clear endpoint."""
    
    def test_clear_embeddings_only(self, client, setup_test_data):
        """Test clearing only embeddings."""
        initial_stats = client.get("/stats").json()
        initial_embeddings = initial_stats["total_embeddings"]
        initial_images = initial_stats["total_images"]
        
        assert initial_embeddings > 0
        
        response = client.delete("/clear?clear_images=false")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert data["embeddings_cleared"] == initial_embeddings
        assert data["images_deleted"] == 0
        assert data["embeddings_remaining"] == 0
        
        # Verify images still exist
        stats = client.get("/stats").json()
        assert stats["total_embeddings"] == 0
        assert stats["total_images"] == initial_images
    
    def test_clear_both_embeddings_and_images(self, client, setup_test_data):
        """Test clearing both embeddings and images."""
        initial_stats = client.get("/stats").json()
        initial_embeddings = initial_stats["total_embeddings"]
        
        response = client.delete("/clear?clear_images=true")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert data["embeddings_cleared"] == initial_embeddings
        assert data["images_deleted"] > 0
        assert data["embeddings_remaining"] == 0
        assert data["images_remaining"] == 0
    
    def test_clear_empty_store(self, client):
        """Test clearing an already empty store."""
        # Ensure store is empty
        client.delete("/clear?clear_images=true")
        
        response = client.delete("/clear")
        assert response.status_code == 200
        
        data = response.json()
        assert data["embeddings_cleared"] == 0


class TestGetDatasetsEndpoint:
    """Tests for the get available datasets endpoint."""
    
    def test_get_datasets_list(self, client):
        """Test getting list of available datasets."""
        response = client.get("/datasets")
        assert response.status_code == 200
        
        data = response.json()
        assert "datasets" in data
        assert "count" in data
        assert data["count"] > 0
        assert isinstance(data["datasets"], dict)
    
    def test_datasets_have_required_fields(self, client):
        """Test that each dataset has all required fields."""
        response = client.get("/datasets")
        assert response.status_code == 200
        
        data = response.json()
        
        for name, info in data["datasets"].items():
            assert "name" in info
            assert "description" in info
            assert "n_classes" in info
            assert "image_size" in info
            assert "channels" in info
            assert info["n_classes"] > 0
    
    def test_datasets_include_common_ones(self, client):
        """Test that common MedMNIST datasets are included."""
        response = client.get("/datasets")
        assert response.status_code == 200
        
        data = response.json()
        datasets = data["datasets"]
        
        common_datasets = ["PathMNIST", "ChestMNIST", "PneumoniaMNIST", "OrganSMNIST"]
        for ds_name in common_datasets:
            assert ds_name in datasets, f"Missing dataset: {ds_name}"


class TestGetLabelsEndpoint:
    """Tests for the get available labels endpoint."""
    
    def test_get_labels(self, client):
        """Test getting available labels."""
        response = client.get("/labels")
        assert response.status_code == 200
        
        data = response.json()
        assert "labels" in data
        assert "count" in data
        assert "dataset" in data
        assert isinstance(data["labels"], dict)
        assert data["count"] > 0
    
    def test_labels_are_strings(self, client):
        """Test that all label values are strings."""
        response = client.get("/labels")
        assert response.status_code == 200
        
        data = response.json()
        
        for value in data["labels"].values():
            assert isinstance(value, str)
            assert len(value) > 0


class TestSearchEndpoint:
    """Extended tests for search endpoint."""
    
    def test_search_with_different_n_results(self, client, setup_test_data, sample_image_base64):
        """Test search with varying n_results values."""
        for n in [1, 3, 5]:
            request = {
                "image_base64": sample_image_base64,
                "n_results": n
            }
            
            response = client.post("/search", json=request)
            assert response.status_code == 200
            
            data = response.json()
            assert data["count"] <= n
            assert len(data["results"]) <= n
    
    def test_search_returns_query_info(self, client, setup_test_data, sample_image_base64):
        """Test that search returns query info."""
        request = {
            "image_base64": sample_image_base64,
            "n_results": 3
        }
        
        response = client.post("/search", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert "query_info" in data
        assert "image_size" in data["query_info"]
        assert "image_mode" in data["query_info"]
    
    def test_search_invalid_base64(self, client):
        """Test search with invalid base64 string."""
        request = {
            "image_base64": "not_a_valid_base64_image",
            "n_results": 5
        }
        
        response = client.post("/search", json=request)
        assert response.status_code == 400


class TestSearchByLabelEndpoint:
    """Extended tests for search by label endpoint."""
    
    def test_search_by_label_with_results(self, client, setup_test_data):
        """Test searching by label when results exist."""
        request = {
            "label": 0,
            "n_results": 10
        }
        
        response = client.post("/search/label", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert "count" in data
        assert "results" in data
        assert isinstance(data["results"], list)
    
    def test_search_by_label_no_results(self, client, setup_test_data):
        """Test searching by label with no matching results."""
        request = {
            "label": 99,  # Non-existent label for testing empty results
            "n_results": 10
        }
        
        response = client.post("/search/label", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["count"] == 0
        assert len(data["results"]) == 0
    
    def test_search_by_label_includes_human_readable(self, client, setup_test_data):
        """Test that search by label includes human readable label."""
        request = {
            "label": 0,
            "n_results": 5
        }
        
        response = client.post("/search/label", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert "human_readable_label" in data["query_info"]
    
    def test_search_by_label_with_n_results_limit(self, client, setup_test_data):
        """Test search by label respects n_results limit."""
        request = {
            "label": 0,
            "n_results": 2
        }
        
        response = client.post("/search/label", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["count"] <= 2


class TestAddImageEndpoint:
    """Extended tests for add image endpoint."""
    
    def test_add_image_with_metadata(self, client, sample_image_base64):
        """Test adding image with custom metadata."""
        request = {
            "image_base64": sample_image_base64,
            "metadata": {
                "label": 5,
                "custom_field": "test_value",
                "number": 42
            }
        }
        
        response = client.post("/add", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "id" in data
        assert "metadata" in data
        assert data["metadata"]["custom_field"] == "test_value"
        assert data["metadata"]["number"] == 42
    
    def test_add_image_without_metadata(self, client, sample_image_base64):
        """Test adding image without metadata."""
        request = {
            "image_base64": sample_image_base64
        }
        
        response = client.post("/add", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
    
    def test_add_image_returns_unique_ids(self, client, sample_image_base64):
        """Test that consecutive additions return unique IDs."""
        ids = []
        
        for _ in range(3):
            request = {"image_base64": sample_image_base64}
            response = client.post("/add", json=request)
            assert response.status_code == 200
            ids.append(response.json()["id"])
        
        # All IDs should be unique
        assert len(ids) == len(set(ids))
    
    def test_add_image_invalid_base64(self, client):
        """Test adding image with invalid base64."""
        request = {
            "image_base64": "invalid_base64_string"
        }
        
        response = client.post("/add", json=request)
        assert response.status_code == 400


class TestResponseModels:
    """Test that API responses match their models."""
    
    def test_search_response_structure(self, client, setup_test_data, sample_image_base64):
        """Test search response has correct structure."""
        request = {
            "image_base64": sample_image_base64,
            "n_results": 3
        }
        
        response = client.post("/search", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert "query_info" in data
        assert "results" in data
        assert "count" in data
        
        for result in data["results"]:
            assert "id" in result
            assert "distance" in result
            assert "metadata" in result
    
    def test_health_response_structure(self, client):
        """Test health response structure."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "encoder_model" in data
        assert "collection_name" in data
        assert "embeddings_count" in data
    
    def test_stats_response_structure(self, client):
        """Test stats response structure."""
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_embeddings" in data
        assert "total_images" in data
        assert "collection_name" in data
        assert "persist_directory" in data
        assert "image_store_directory" in data
