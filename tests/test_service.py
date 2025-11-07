"""Tests for FastAPI service module."""

import pytest
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from vision_rag.service import app
from vision_rag.encoder import CLIPImageEncoder
from vision_rag.rag_store import ChromaRAGStore
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
            data="not valid json",
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
