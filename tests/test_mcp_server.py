"""Tests for MCP server module."""

import asyncio
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from vision_rag.mcp_server import VisionRAGMCPServer
from vision_rag.utils import encode_image_to_base64


@pytest.fixture
def mcp_server():
    """Create an MCP server instance for testing."""
    server = VisionRAGMCPServer(
        collection_name="test_mcp_collection",
        persist_directory="./chroma_db_test_mcp",
    )
    
    yield server
    
    # Cleanup after tests
    server.rag_store.clear()


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
    return Image.fromarray(img_array, mode='L')


@pytest.fixture
def sample_image_base64(sample_image):
    """Create a base64 encoded sample image."""
    return encode_image_to_base64(sample_image)


@pytest.fixture
async def server_with_data(mcp_server, sample_image_base64):
    """Setup MCP server with test data."""
    # Add several test images
    for i in range(5):
        img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        
        await mcp_server.add_image(
            image_base64=encode_image_to_base64(img),
            metadata={"label": i % 3, "test": True, "index": i}
        )
    
    yield mcp_server
    
    # Cleanup
    mcp_server.rag_store.clear()


class TestMCPServerInitialization:
    """Tests for MCP server initialization."""
    
    def test_server_initialization(self, mcp_server):
        """Test server initializes correctly."""
        assert mcp_server.encoder is not None
        assert mcp_server.rag_store is not None
        assert mcp_server.searcher is not None
        assert len(mcp_server.tools) == 9
        
        # Check all expected tools are registered
        expected_tools = [
            "search_similar_images",
            "search_by_label",
            "add_image",
            "get_statistics",
            "list_available_labels",
        ]
        for tool in expected_tools:
            assert tool in mcp_server.tools
    
    def test_server_components(self, mcp_server):
        """Test server components are properly initialized."""
        # Check encoder
        assert hasattr(mcp_server.encoder, 'encode_image')
        assert hasattr(mcp_server.encoder, 'encode_images')
        
        # Check RAG store
        assert hasattr(mcp_server.rag_store, 'add_embeddings')
        assert hasattr(mcp_server.rag_store, 'search')
        assert hasattr(mcp_server.rag_store, 'count')
        
        # Check searcher
        assert hasattr(mcp_server.searcher, 'search')
    
    def test_server_with_custom_image_size(self):
        """Test server initialization with custom image size."""
        server = VisionRAGMCPServer(
            collection_name="test_custom_size",
            persist_directory="./chroma_db_test_custom",
            image_size=128
        )
        
        assert server.image_store is not None
        
        # Cleanup
        server.rag_store.clear()
    
    def test_server_with_none_image_size(self):
        """Test server initialization with image_size=None (no resizing)."""
        server = VisionRAGMCPServer(
            collection_name="test_no_resize",
            persist_directory="./chroma_db_test_no_resize",
            image_size=None
        )
        
        assert server.image_store is not None
        
        # Cleanup
        server.rag_store.clear()
    
    def test_server_custom_directories(self):
        """Test server with custom directories."""
        server = VisionRAGMCPServer(
            collection_name="test_custom_dirs",
            persist_directory="chroma_db_test_mcp_server",
            image_store_dir="./custom_images"
        )
        
        assert server.rag_store.collection_name == "test_custom_dirs"
        assert server.rag_store.persist_directory == "chroma_db_test_mcp_server"
        
        # Cleanup
        server.rag_store.clear()


class TestHandleToolCall:
    """Tests for the handle_tool_call method."""
    
    @pytest.mark.asyncio
    async def test_handle_valid_tool_call(self, mcp_server):
        """Test handling a valid tool call."""
        result = await mcp_server.handle_tool_call(
            "get_statistics",
            {}
        )
        
        assert result["success"] is True
        assert "result" in result
        assert "total_embeddings" in result["result"]
    
    @pytest.mark.asyncio
    async def test_handle_unknown_tool(self, mcp_server):
        """Test handling unknown tool call."""
        result = await mcp_server.handle_tool_call(
            "unknown_tool",
            {}
        )
        
        assert "error" in result
        assert "Unknown tool" in result["error"]
        assert "available_tools" in result
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_with_error(self, mcp_server):
        """Test handling tool call that raises an error."""
        # Call search with invalid base64
        result = await mcp_server.handle_tool_call(
            "search_similar_images",
            {"image_base64": "invalid_base64"}
        )
        
        assert result["success"] is False
        assert "error" in result


class TestSearchSimilarImages:
    """Tests for the search_similar_images tool."""
    
    @pytest.mark.asyncio
    async def test_search_similar_images_empty_store(self, mcp_server, sample_image_base64):
        """Test searching in an empty store."""
        result = await mcp_server.search_similar_images(
            image_base64=sample_image_base64,
            n_results=5
        )
        
        assert "results" in result
        assert "count" in result
        assert result["count"] == 0
        assert len(result["results"]) == 0
        assert "query_info" in result
    
    @pytest.mark.asyncio
    async def test_search_similar_images_with_data(self, server_with_data, sample_image_base64):
        """Test searching with data in store."""
        result = await server_with_data.search_similar_images(
            image_base64=sample_image_base64,
            n_results=3
        )
        
        assert "results" in result
        assert "count" in result
        assert result["count"] <= 3
        assert len(result["results"]) <= 3
        
        # Check query info
        assert "query_info" in result
        assert "image_size" in result["query_info"]
        assert "image_mode" in result["query_info"]
        
        # Check result structure
        if result["count"] > 0:
            first_result = result["results"][0]
            assert "id" in first_result
            assert "distance" in first_result
            assert "metadata" in first_result
    
    @pytest.mark.asyncio
    async def test_search_includes_human_readable_labels(self, server_with_data, sample_image_base64):
        """Test that search results include human-readable labels."""
        result = await server_with_data.search_similar_images(
            image_base64=sample_image_base64,
            n_results=5
        )
        
        # Check if results with labels have human-readable labels
        for item in result["results"]:
            if "label" in item["metadata"]:
                assert "human_readable_label" in item
    
    @pytest.mark.asyncio
    async def test_search_n_results_parameter(self, server_with_data, sample_image_base64):
        """Test n_results parameter limits results."""
        result = await server_with_data.search_similar_images(
            image_base64=sample_image_base64,
            n_results=2
        )
        
        assert result["count"] <= 2
        assert len(result["results"]) <= 2
    
    @pytest.mark.asyncio
    async def test_search_invalid_base64(self, mcp_server):
        """Test search with invalid base64 raises error."""
        with pytest.raises(Exception):
            await mcp_server.search_similar_images(
                image_base64="not_valid_base64",
                n_results=5
            )


class TestSearchByLabel:
    """Tests for the search_by_label tool."""
    
    @pytest.mark.asyncio
    async def test_search_by_label_with_matches(self, server_with_data):
        """Test searching by label with matches."""
        result = await server_with_data.search_by_label(label=0)
        
        assert "label" in result
        assert result["label"] == 0
        assert "human_readable_label" in result
        assert "ids" in result
        assert "metadatas" in result
        assert "count" in result
        
        # Should have matches for label 0
        assert result["count"] > 0
        
        # All results should have the requested label
        for metadata in result["metadatas"]:
            assert metadata["label"] == 0
    
    @pytest.mark.asyncio
    async def test_search_by_label_no_matches(self, server_with_data):
        """Test searching by label with no matches."""
        result = await server_with_data.search_by_label(label=999)
        
        assert result["label"] == 999
        assert result["count"] == 0
        assert len(result["ids"]) == 0
    
    @pytest.mark.asyncio
    async def test_search_by_label_with_limit(self, server_with_data):
        """Test search by label with result limit."""
        result = await server_with_data.search_by_label(label=0, n_results=1)
        
        assert result["count"] <= 1
        assert len(result["ids"]) <= 1
    
    @pytest.mark.asyncio
    async def test_search_by_label_empty_store(self, mcp_server):
        """Test search by label in empty store."""
        result = await mcp_server.search_by_label(label=0)
        
        assert result["count"] == 0
        assert len(result["ids"]) == 0
    
    @pytest.mark.asyncio
    async def test_search_by_label_with_return_images(self, server_with_data):
        """Test search by label with return_images=True."""
        result = await server_with_data.search_by_label(label=0, return_images=True)
        
        assert "images_base64" in result
        assert isinstance(result["images_base64"], list)
        assert len(result["images_base64"]) == result["count"]
        
        # Check that images are base64 strings
        for img_b64 in result["images_base64"]:
            if img_b64 is not None:
                assert isinstance(img_b64, str)
                assert len(img_b64) > 0
    
    @pytest.mark.asyncio
    async def test_search_by_label_without_return_images(self, server_with_data):
        """Test search by label with return_images=False (default)."""
        result = await server_with_data.search_by_label(label=0, return_images=False)
        
        # Should not include images_base64
        assert "images_base64" not in result
        assert "ids" in result
        assert "metadatas" in result
    
    @pytest.mark.asyncio
    async def test_search_by_label_return_images_missing_path(self, mcp_server, sample_image_base64):
        """Test search by label with return_images when image_path is missing."""
        # Add image without image_path in metadata (manually)
        image = Image.fromarray(np.random.randint(0, 255, size=(28, 28), dtype=np.uint8), mode='L')
        embedding = mcp_server.encoder.encode_image(image)
        
        # Add directly to rag_store without using add_image (so no image_path)
        mcp_server.rag_store.add_embeddings(
            embeddings=embedding.reshape(1, -1),
            ids=["test_no_path"],
            metadatas=[{"label": 5}]
        )
        
        result = await mcp_server.search_by_label(label=5, return_images=True)
        
        assert "images_base64" in result
        # Should have None for missing images
        assert None in result["images_base64"]
    
    @pytest.mark.asyncio
    async def test_search_by_label_return_images_corrupted(self, mcp_server, sample_image_base64, tmp_path):
        """Test search by label with return_images when image file is corrupted."""
        # Add image normally to get an entry in the database
        image = Image.fromarray(np.random.randint(0, 255, size=(28, 28), dtype=np.uint8), mode='L')
        embedding = mcp_server.encoder.encode_image(image)
        
        # Create a corrupted image file
        corrupted_path = tmp_path / "corrupted_image.png"
        corrupted_path.write_text("This is not a valid image file")
        
        # Add directly to rag_store with path to corrupted file
        mcp_server.rag_store.add_embeddings(
            embeddings=embedding.reshape(1, -1),
            ids=["test_corrupted"],
            metadatas=[{"label": 7, "image_path": str(corrupted_path)}]
        )
        
        result = await mcp_server.search_by_label(label=7, return_images=True)
        
        assert "images_base64" in result
        # Should have None for corrupted images
        assert None in result["images_base64"]
    
    @pytest.mark.asyncio
    async def test_search_by_label_return_images_limit(self, server_with_data):
        """Test search by label with both return_images and n_results limit."""
        result = await server_with_data.search_by_label(label=0, n_results=1, return_images=True)
        
        assert result["count"] <= 1
        assert "images_base64" in result
        assert len(result["images_base64"]) <= 1


class TestAddImage:
    """Tests for the add_image tool."""
    
    @pytest.mark.asyncio
    async def test_add_image_basic(self, mcp_server, sample_image_base64):
        """Test adding a single image."""
        initial_count = mcp_server.rag_store.count()
        
        result = await mcp_server.add_image(
            image_base64=sample_image_base64,
            metadata={"label": 5, "source": "test"}
        )
        
        assert "id" in result
        assert "metadata" in result
        assert "image_path" in result
        assert "total_embeddings" in result
        assert result["metadata"]["label"] == 5
        assert result["metadata"]["source"] == "test"
        assert result["total_embeddings"] == initial_count + 1
        
        # Verify image was saved to disk
        assert "image_path" in result["metadata"]
        image_path = Path(result["image_path"])
        assert image_path.exists()
    
    @pytest.mark.asyncio
    async def test_add_image_without_metadata(self, mcp_server, sample_image_base64):
        """Test adding image without metadata."""
        result = await mcp_server.add_image(
            image_base64=sample_image_base64
        )
        
        assert "id" in result
        assert "metadata" in result
        assert "index" in result["metadata"]
        assert "image_path" in result["metadata"]
    
    @pytest.mark.asyncio
    async def test_add_multiple_images(self, mcp_server):
        """Test adding multiple images sequentially."""
        images_to_add = 3
        
        for i in range(images_to_add):
            img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L')
            
            result = await mcp_server.add_image(
                image_base64=encode_image_to_base64(img),
                metadata={"label": i}
            )
            
            assert result["total_embeddings"] == i + 1
    
    @pytest.mark.asyncio
    async def test_add_image_invalid_base64(self, mcp_server):
        """Test adding image with invalid base64."""
        with pytest.raises(Exception):
            await mcp_server.add_image(
                image_base64="invalid_base64_data"
            )
    
    @pytest.mark.asyncio
    async def test_add_image_with_rgb(self, mcp_server):
        """Test adding RGB image."""
        # Create RGB image
        img_array = np.random.randint(0, 255, size=(28, 28, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')
        
        result = await mcp_server.add_image(
            image_base64=encode_image_to_base64(img),
            metadata={"type": "rgb"}
        )
        
        assert "id" in result
        assert "image_path" in result
        
        # Verify saved image
        saved_path = Path(result["image_path"])
        assert saved_path.exists()
    
    @pytest.mark.asyncio
    async def test_add_image_metadata_preserved(self, mcp_server, sample_image_base64):
        """Test that metadata is properly preserved."""
        metadata = {
            "label": 3,
            "patient_id": "12345",
            "scan_date": "2025-01-01",
            "custom_field": "test_value"
        }
        
        result = await mcp_server.add_image(
            image_base64=sample_image_base64,
            metadata=metadata
        )
        
        # Check all metadata fields are preserved
        for key, value in metadata.items():
            assert result["metadata"][key] == value
        
        # Also includes image_path and index
        assert "image_path" in result["metadata"]
        assert "index" in result["metadata"]


class TestGetStatistics:
    """Tests for the get_statistics tool."""
    
    @pytest.mark.asyncio
    async def test_get_statistics_empty(self, mcp_server):
        """Test getting statistics from empty store."""
        result = await mcp_server.get_statistics()
        
        assert "total_embeddings" in result
        assert "total_images" in result
        assert "collection_name" in result
        assert "persist_directory" in result
        assert "image_store_directory" in result
        assert "encoder_model" in result
        assert "embedding_dimension" in result
        
        assert isinstance(result["total_embeddings"], int)
        assert isinstance(result["total_images"], int)
        assert isinstance(result["embedding_dimension"], int)
    
    @pytest.mark.asyncio
    async def test_get_statistics_with_data(self, server_with_data):
        """Test getting statistics with data in store."""
        result = await server_with_data.get_statistics()
        
        assert result["total_embeddings"] == 5
        # Note: total_images may be >= 5 due to shared image store directory
        assert result["total_images"] >= 5
        assert result["collection_name"] == "test_mcp_collection"
        assert result["embedding_dimension"] > 0
    
    @pytest.mark.asyncio
    async def test_statistics_fields_consistency(self, mcp_server):
        """Test that statistics fields are consistent."""
        result = await mcp_server.get_statistics()
        
        # Verify all expected fields are present
        expected_fields = [
            "total_embeddings",
            "total_images",
            "collection_name",
            "persist_directory",
            "image_store_directory",
            "encoder_model",
            "embedding_dimension"
        ]
        
        for field in expected_fields:
            assert field in result, f"Missing field: {field}"


class TestListAvailableLabels:
    """Tests for the list_available_labels tool."""
    
    @pytest.mark.asyncio
    async def test_list_available_labels(self, mcp_server):
        """Test listing available labels."""
        result = await mcp_server.list_available_labels()
        
        assert "labels" in result
        assert "count" in result
        assert isinstance(result["labels"], dict)
        assert result["count"] == len(result["labels"])
        
        # Check label structure
        for label_id, label_name in result["labels"].items():
            assert isinstance(label_id, int)
            assert isinstance(label_name, str)


class TestGetToolDefinitions:
    """Tests for the get_tool_definitions method."""
    
    def test_get_tool_definitions(self, mcp_server):
        """Test getting tool definitions."""
        definitions = mcp_server.get_tool_definitions()
        
        assert isinstance(definitions, list)
        assert len(definitions) == 9
        
        # Check structure of each definition
        for tool_def in definitions:
            assert "name" in tool_def
            assert "description" in tool_def
            assert "parameters" in tool_def
            assert isinstance(tool_def["parameters"], dict)
    
    def test_tool_definitions_completeness(self, mcp_server):
        """Test that all tools have definitions."""
        definitions = mcp_server.get_tool_definitions()
        defined_tool_names = {tool["name"] for tool in definitions}
        
        for tool_name in mcp_server.tools.keys():
            assert tool_name in defined_tool_names
    
    def test_search_similar_images_definition(self, mcp_server):
        """Test search_similar_images tool definition."""
        definitions = mcp_server.get_tool_definitions()
        search_def = next(d for d in definitions if d["name"] == "search_similar_images")
        
        assert "image_base64" in search_def["parameters"]
        assert "n_results" in search_def["parameters"]
        assert "required" in search_def
        assert "image_base64" in search_def["required"]
        assert "n_results" not in search_def["required"]
    
    def test_search_by_label_definition(self, mcp_server):
        """Test search_by_label tool definition includes return_images."""
        definitions = mcp_server.get_tool_definitions()
        label_search_def = next(d for d in definitions if d["name"] == "search_by_label")
        
        assert "label" in label_search_def["parameters"]
        assert "n_results" in label_search_def["parameters"]
        assert "return_images" in label_search_def["parameters"]
        assert "required" in label_search_def
        assert "label" in label_search_def["required"]
        assert label_search_def["parameters"]["return_images"]["default"] is False
    
    def test_add_image_definition(self, mcp_server):
        """Test add_image tool definition."""
        definitions = mcp_server.get_tool_definitions()
        add_def = next(d for d in definitions if d["name"] == "add_image")
        
        assert "image_base64" in add_def["parameters"]
        assert "metadata" in add_def["parameters"]
        assert "required" in add_def
        assert "image_base64" in add_def["required"]
        assert "metadata" not in add_def["required"]
    
    def test_tool_definition_types(self, mcp_server):
        """Test that tool definitions have correct parameter types."""
        definitions = mcp_server.get_tool_definitions()
        
        for tool_def in definitions:
            params = tool_def["parameters"]
            for param_name, param_info in params.items():
                assert "type" in param_info
                assert "description" in param_info
            # Check that required field exists at tool level
            assert "required" in tool_def


class TestEndToEndWorkflow:
    """Test complete workflow with MCP server."""
    
    @pytest.mark.asyncio
    async def test_complete_mcp_workflow(self, mcp_server):
        """Test complete workflow: add, search, stats."""
        # 1. Get initial statistics
        stats = await mcp_server.get_statistics()
        initial_count = stats["total_embeddings"]
        
        # 2. Add images
        images = []
        for i in range(3):
            img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L')
            images.append(img)
            
            result = await mcp_server.add_image(
                image_base64=encode_image_to_base64(img),
                metadata={"label": i % 2, "index": i}
            )
            assert "id" in result
        
        # 3. Verify statistics updated
        stats = await mcp_server.get_statistics()
        assert stats["total_embeddings"] == initial_count + 3
        
        # 4. Search for similar images
        search_result = await mcp_server.search_similar_images(
            image_base64=encode_image_to_base64(images[0]),
            n_results=2
        )
        assert search_result["count"] <= 2
        
        # 5. Search by label
        label_result = await mcp_server.search_by_label(label=0)
        assert label_result["count"] >= 1
        
        # 6. List labels
        labels = await mcp_server.list_available_labels()
        assert labels["count"] > 0
    
    @pytest.mark.asyncio
    async def test_tool_call_integration(self, mcp_server, sample_image_base64):
        """Test using handle_tool_call for all operations."""
        # Add image via handle_tool_call
        add_result = await mcp_server.handle_tool_call(
            "add_image",
            {"image_base64": sample_image_base64, "metadata": {"label": 1}}
        )
        assert add_result["success"] is True
        
        # Get statistics via handle_tool_call
        stats_result = await mcp_server.handle_tool_call(
            "get_statistics",
            {}
        )
        assert stats_result["success"] is True
        assert stats_result["result"]["total_embeddings"] >= 1
        
        # Search via handle_tool_call
        search_result = await mcp_server.handle_tool_call(
            "search_similar_images",
            {"image_base64": sample_image_base64, "n_results": 5}
        )
        assert search_result["success"] is True
        
        # List labels via handle_tool_call
        labels_result = await mcp_server.handle_tool_call(
            "list_available_labels",
            {}
        )
        assert labels_result["success"] is True


class TestErrorHandling:
    """Test error handling in MCP server."""
    
    @pytest.mark.asyncio
    async def test_search_with_corrupted_data(self, mcp_server):
        """Test search with corrupted base64 data."""
        with pytest.raises(Exception):
            await mcp_server.search_similar_images(
                image_base64="corrupted!!!data",
                n_results=5
            )
    
    @pytest.mark.asyncio
    async def test_add_with_invalid_metadata_type(self, mcp_server, sample_image_base64):
        """Test add_image with metadata still works (metadata is flexible)."""
        # MCP server should handle various metadata types
        result = await mcp_server.add_image(
            image_base64=sample_image_base64,
            metadata={"custom_field": "test_value", "number": 42}
        )
        assert "id" in result
        assert result["metadata"]["custom_field"] == "test_value"
    
    @pytest.mark.asyncio
    async def test_search_with_negative_n_results(self, mcp_server, sample_image_base64):
        """Test search with negative n_results (ChromaDB should handle this)."""
        # ChromaDB may handle negative values differently
        # This tests that the server doesn't crash
        try:
            result = await mcp_server.search_similar_images(
                image_base64=sample_image_base64,
                n_results=-1
            )
            # If it succeeds, check the result
            assert "results" in result
        except Exception:
            # If it raises an exception, that's also acceptable
            pass
    
    @pytest.mark.asyncio
    async def test_search_with_very_large_n_results(self, server_with_data, sample_image_base64):
        """Test search with very large n_results."""
        result = await server_with_data.search_similar_images(
            image_base64=sample_image_base64,
            n_results=1000
        )
        
        # Should return at most the number of items in store
        assert result["count"] <= 5


class TestImageSizeIntegration:
    """Test image size parameter integration."""
    
    @pytest.mark.asyncio
    async def test_add_image_with_resizing(self):
        """Test adding images with resizing enabled."""
        server = VisionRAGMCPServer(
            collection_name="test_resize",
            persist_directory="./chroma_db_test_resize",
            image_size=64
        )
        
        # Create a larger image
        img_array = np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        
        result = await server.add_image(
            image_base64=encode_image_to_base64(img),
            metadata={"original_size": "128x128"}
        )
        
        # Verify image was saved
        saved_path = Path(result["image_path"])
        assert saved_path.exists()
        
        # Verify image was resized
        saved_img = Image.open(saved_path)
        assert saved_img.size == (64, 64)
        
        # Cleanup
        server.rag_store.clear()
    
    @pytest.mark.asyncio
    async def test_add_image_without_resizing(self):
        """Test adding images without resizing (image_size=None)."""
        server = VisionRAGMCPServer(
            collection_name="test_no_resize",
            persist_directory="./chroma_db_test_no_resize",
            image_size=None
        )
        
        # Create an image
        original_size = (100, 100)
        img_array = np.random.randint(0, 255, size=original_size, dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        
        result = await server.add_image(
            image_base64=encode_image_to_base64(img),
            metadata={"test": "no_resize"}
        )
        
        # Verify image was saved at original size
        saved_path = Path(result["image_path"])
        assert saved_path.exists()
        
        saved_img = Image.open(saved_path)
        assert saved_img.size == original_size
        
        # Cleanup
        server.rag_store.clear()


class TestConcurrentOperations:
    """Test concurrent operations on MCP server."""
    
    @pytest.mark.asyncio
    async def test_concurrent_add_images(self, mcp_server):
        """Test adding multiple images concurrently."""
        
        async def add_test_image(index):
            img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L')
            return await mcp_server.add_image(
                image_base64=encode_image_to_base64(img),
                metadata={"index": index}
            )
        
        # Add 5 images concurrently
        tasks = [add_test_image(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # Verify all succeeded
        assert len(results) == 5
        for result in results:
            assert "id" in result
            assert "metadata" in result
        
        # Verify all are in the store
        stats = await mcp_server.get_statistics()
        assert stats["total_embeddings"] >= 5
    
    @pytest.mark.asyncio
    async def test_concurrent_searches(self, server_with_data):
        """Test multiple concurrent searches."""
        
        
        async def search_test():
            img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L')
            return await server_with_data.search_similar_images(
                image_base64=encode_image_to_base64(img),
                n_results=3
            )
        
        # Perform 5 searches concurrently
        tasks = [search_test() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # Verify all succeeded
        assert len(results) == 5
        for result in results:
            assert "results" in result
            assert "count" in result


class TestSearchEdgeCases:
    """Test edge cases in search functionality."""
    
    @pytest.mark.asyncio
    async def test_search_with_empty_image(self, mcp_server):
        """Test searching with a very small image."""
        # Create small image (must be at least a few pixels for CLIP)
        img = Image.fromarray(np.full((8, 8), 128, dtype=np.uint8), mode='L')
        
        result = await mcp_server.search_similar_images(
            image_base64=encode_image_to_base64(img),
            n_results=1
        )
        
        assert "query_info" in result
        assert result["query_info"]["image_size"] == [8, 8]
    
    @pytest.mark.asyncio
    async def test_search_with_zero_results(self, server_with_data):
        """Test search with n_results=1 (ChromaDB doesn't allow 0)."""
        img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        
        result = await server_with_data.search_similar_images(
            image_base64=encode_image_to_base64(img),
            n_results=1
        )
        
        assert result["count"] >= 0
        assert "results" in result
    
    @pytest.mark.asyncio
    async def test_search_similar_different_image_modes(self, server_with_data):
        """Test searching with RGB image when store has grayscale."""
        # Create RGB image
        img_array = np.random.randint(0, 255, size=(28, 28, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        result = await server_with_data.search_similar_images(
            image_base64=encode_image_to_base64(img),
            n_results=3
        )
        
        assert "query_info" in result
        assert result["query_info"]["image_mode"] == "RGB"
        assert "results" in result


class TestSearchByLabelEdgeCases:
    """Test edge cases for search by label."""
    
    @pytest.mark.asyncio
    async def test_search_by_label_boundary_values(self, server_with_data):
        """Test search with boundary label values (0 and 10)."""
        # Test label 0
        result_0 = await server_with_data.search_by_label(label=0)
        assert result_0["label"] == 0
        assert "human_readable_label" in result_0
        
        # Test label 10
        result_10 = await server_with_data.search_by_label(label=10)
        assert result_10["label"] == 10
        assert "human_readable_label" in result_10
    
    @pytest.mark.asyncio
    async def test_search_by_label_return_images_empty_results(self, mcp_server):
        """Test return_images=True when no results found."""
        result = await mcp_server.search_by_label(
            label=999,
            return_images=True
        )
        
        assert result["count"] == 0
        assert result["images_base64"] == []
    
    @pytest.mark.asyncio
    async def test_search_by_label_with_unlimited_results(self, server_with_data):
        """Test search_by_label with n_results=None."""
        result = await server_with_data.search_by_label(
            label=0,
            n_results=None
        )
        
        # Should return all matches
        assert "count" in result
        assert result["count"] >= 0


class TestAddImageEdgeCases:
    """Test edge cases for adding images."""
    
    @pytest.mark.asyncio
    async def test_add_image_with_empty_metadata(self, mcp_server):
        """Test adding image with explicitly empty metadata dict."""
        img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        
        result = await mcp_server.add_image(
            image_base64=encode_image_to_base64(img),
            metadata={}
        )
        
        assert "id" in result
        assert "metadata" in result
        assert "image_path" in result["metadata"]
        assert "index" in result["metadata"]
    
    @pytest.mark.asyncio
    async def test_add_image_with_complex_metadata(self, mcp_server):
        """Test adding image with flat metadata (ChromaDB doesn't support nested dicts)."""
        img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        
        # ChromaDB only supports flat metadata with basic types
        metadata = {
            "label": 5,
            "patient_id": "P12345",
            "patient_age": 45,
            "scan_date": "2025-11-10",
            "modality": "CT"
        }
        
        result = await mcp_server.add_image(
            image_base64=encode_image_to_base64(img),
            metadata=metadata
        )
        
        assert "metadata" in result
        assert result["metadata"]["label"] == 5
        assert result["metadata"]["patient_id"] == "P12345"
        assert result["metadata"]["patient_age"] == 45
    
    @pytest.mark.asyncio
    async def test_add_image_preserves_id_sequence(self, mcp_server):
        """Test that image IDs are sequential."""
        ids = []
        
        for i in range(3):
            img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='L')
            result = await mcp_server.add_image(
                image_base64=encode_image_to_base64(img)
            )
            ids.append(result["id"])
        
        # Check IDs are sequential
        assert ids[0] == "img_0"
        assert ids[1] == "img_1"
        assert ids[2] == "img_2"
    
    @pytest.mark.asyncio
    async def test_add_large_image(self, mcp_server):
        """Test adding a large image."""
        # Create a large image
        img_array = np.random.randint(0, 255, size=(512, 512), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        
        result = await mcp_server.add_image(
            image_base64=encode_image_to_base64(img),
            metadata={"size": "large"}
        )
        
        assert "id" in result
        assert result["metadata"]["size"] == "large"


class TestStatisticsEdgeCases:
    """Test edge cases for statistics."""
    
    @pytest.mark.asyncio
    async def test_statistics_after_multiple_operations(self, mcp_server):
        """Test that statistics update correctly after multiple operations."""
        # Get initial stats
        stats1 = await mcp_server.get_statistics()
        initial_count = stats1["total_embeddings"]
        
        # Add an image
        img_array = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        await mcp_server.add_image(image_base64=encode_image_to_base64(img))
        
        # Check stats updated
        stats2 = await mcp_server.get_statistics()
        assert stats2["total_embeddings"] == initial_count + 1
        
        # Add another
        await mcp_server.add_image(image_base64=encode_image_to_base64(img))
        stats3 = await mcp_server.get_statistics()
        assert stats3["total_embeddings"] == initial_count + 2
    
    @pytest.mark.asyncio
    async def test_statistics_field_types(self, mcp_server):
        """Test that all statistics fields have correct types."""
        stats = await mcp_server.get_statistics()
        
        assert isinstance(stats["total_embeddings"], int)
        assert isinstance(stats["total_images"], int)
        assert isinstance(stats["collection_name"], str)
        assert isinstance(stats["persist_directory"], str)
        assert isinstance(stats["image_store_directory"], str)
        assert isinstance(stats["encoder_model"], str)
        assert isinstance(stats["embedding_dimension"], int)


class TestToolsRegistration:
    """Test tool registration and availability."""
    
    @pytest.mark.asyncio
    async def test_all_tools_callable(self, mcp_server):
        """Test that all registered tools are callable."""
        for tool_name, tool_func in mcp_server.tools.items():
            assert callable(tool_func)
            assert asyncio.iscoroutinefunction(tool_func)
    
    @pytest.mark.asyncio
    async def test_tools_dict_immutable(self, mcp_server):
        """Test that tools dictionary contains expected tools."""
        expected_tools = {
            "search_similar_images",
            "search_by_label",
            "add_image",
            "get_statistics",
            "list_available_labels",
            "list_available_datasets",
            "preload_dataset",
            "clear_store",
            "reindex_from_images",
        }
        
        assert set(mcp_server.tools.keys()) == expected_tools


class TestHandleToolCallEdgeCases:
    """Test edge cases for handle_tool_call."""
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_empty_arguments(self, mcp_server):
        """Test tool call with empty arguments dict."""
        result = await mcp_server.handle_tool_call("get_statistics", {})
        
        assert result["success"] is True
        assert "result" in result
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_unknown_provides_available_tools(self, mcp_server):
        """Test that unknown tool error includes available tools."""
        result = await mcp_server.handle_tool_call("nonexistent_tool", {})
        
        assert "error" in result
        assert "available_tools" in result
        assert len(result["available_tools"]) == 9
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_exception_handling(self, mcp_server):
        """Test that exceptions in tool calls are handled gracefully."""
        # Try to call a tool with invalid arguments that will cause an exception
        result = await mcp_server.handle_tool_call(
            "search_similar_images",
            {"image_base64": "invalid_base64!!!"}
        )
        
        assert result["success"] is False
        assert "error" in result
        assert isinstance(result["error"], str)


class TestToolDefinitionsEdgeCases:
    """Test edge cases for tool definitions."""
    
    @pytest.mark.asyncio
    async def test_tool_definitions_have_descriptions(self, mcp_server):
        """Test that all tool definitions have descriptions."""
        tool_defs = mcp_server.get_tool_definitions()
        
        for tool_def in tool_defs:
            assert "description" in tool_def
            assert len(tool_def["description"]) > 0
            assert isinstance(tool_def["description"], str)
    
    @pytest.mark.asyncio
    async def test_tool_definitions_parameter_descriptions(self, mcp_server):
        """Test that all parameters have descriptions."""
        tool_defs = mcp_server.get_tool_definitions()
        
        for tool_def in tool_defs:
            if "parameters" in tool_def and tool_def["parameters"]:
                for param_name, param_spec in tool_def["parameters"].items():
                    if isinstance(param_spec, dict):
                        assert "description" in param_spec or "type" in param_spec
    
    @pytest.mark.asyncio
    async def test_search_similar_images_has_required_params(self, mcp_server):
        """Test that search_similar_images defines required parameters."""
        tool_defs = mcp_server.get_tool_definitions()
        search_def = next(td for td in tool_defs if td["name"] == "search_similar_images")
        
        assert "required" in search_def
        assert "image_base64" in search_def["required"]


class TestMCPToolSchemaConversion:
    """Test MCP Tool schema generation and conversion."""
    
    @pytest.mark.asyncio
    async def test_tool_definition_to_mcp_schema(self, mcp_server):
        """Test that tool definitions can be converted to MCP Tool schema."""
        tool_defs = mcp_server.get_tool_definitions()
        
        # Simulate what main() does - convert to MCP Tool format
        for tool_def in tool_defs:
            input_schema = {
                "type": "object",
                "properties": tool_def["parameters"],
            }
            if "required" in tool_def and tool_def["required"]:
                input_schema["required"] = tool_def["required"]
            
            # Verify schema structure
            assert "type" in input_schema
            assert input_schema["type"] == "object"
            assert "properties" in input_schema
    
    @pytest.mark.asyncio
    async def test_all_tools_have_valid_schemas(self, mcp_server):
        """Test that all tool schemas are valid JSON Schema objects."""
        tool_defs = mcp_server.get_tool_definitions()
        
        for tool_def in tool_defs:
            assert "name" in tool_def
            assert "description" in tool_def
            assert "parameters" in tool_def
            
            # Parameters should be a dict
            assert isinstance(tool_def["parameters"], dict)
    
    @pytest.mark.asyncio
    async def test_get_statistics_empty_schema(self, mcp_server):
        """Test that get_statistics has empty parameter schema."""
        tool_defs = mcp_server.get_tool_definitions()
        stats_def = next(td for td in tool_defs if td["name"] == "get_statistics")
        
        # Should have empty or no parameters
        assert len(stats_def["parameters"]) == 0
        assert stats_def["required"] == []
    
    @pytest.mark.asyncio
    async def test_list_available_labels_empty_schema(self, mcp_server):
        """Test that list_available_labels has empty parameter schema."""
        tool_defs = mcp_server.get_tool_definitions()
        labels_def = next(td for td in tool_defs if td["name"] == "list_available_labels")
        
        # Should have empty or no parameters
        assert len(labels_def["parameters"]) == 0
        assert labels_def["required"] == []


class TestServerInitializationEdgeCases:
    """Test edge cases in server initialization."""
    
    @pytest.mark.asyncio
    async def test_server_with_default_parameters(self):
        """Test server initialization with all default parameters."""
        server = VisionRAGMCPServer()
        
        assert server.encoder is not None
        assert server.rag_store is not None
        assert server.image_store is not None
        assert server.searcher is not None
        assert len(server.tools) == 9
    
    @pytest.mark.asyncio
    async def test_server_encoder_dimension_accessible(self, mcp_server):
        """Test that encoder dimension is accessible."""
        dim = mcp_server.encoder.embedding_dimension
        assert isinstance(dim, int)
        assert dim > 0
    
    @pytest.mark.asyncio
    async def test_server_components_properly_connected(self, mcp_server):
        """Test that server components are properly connected."""
        # Searcher should use the same encoder and rag_store
        assert mcp_server.searcher.encoder == mcp_server.encoder
        assert mcp_server.searcher.rag_store == mcp_server.rag_store


class TestSearchSimilarImagesEdgeCases:
    """Additional edge cases for search_similar_images."""
    
    @pytest.mark.asyncio
    async def test_search_returns_query_info_for_all_modes(self, mcp_server):
        """Test that query_info is returned for different image modes."""
        # Test grayscale
        gray_img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8), mode='L')
        result_gray = await mcp_server.search_similar_images(
            image_base64=encode_image_to_base64(gray_img),
            n_results=1
        )
        assert result_gray["query_info"]["image_mode"] == "L"
        
        # Test RGB
        rgb_img = Image.fromarray(np.random.randint(0, 255, (28, 28, 3), dtype=np.uint8))
        result_rgb = await mcp_server.search_similar_images(
            image_base64=encode_image_to_base64(rgb_img),
            n_results=1
        )
        assert result_rgb["query_info"]["image_mode"] == "RGB"
    
    @pytest.mark.asyncio
    async def test_search_with_maximum_n_results(self, server_with_data):
        """Test search with very large n_results doesn't break."""
        img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8), mode='L')
        
        result = await server_with_data.search_similar_images(
            image_base64=encode_image_to_base64(img),
            n_results=1000
        )
        
        # Should not crash, just return what's available
        assert "results" in result
        assert result["count"] <= 1000


class TestAddImageEdgeCasesExtended:
    """Extended edge cases for add_image functionality."""
    
    @pytest.mark.asyncio
    async def test_add_image_returns_correct_index(self, mcp_server):
        """Test that added images get correct index in metadata."""
        initial_count = mcp_server.rag_store.count()
        
        img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8), mode='L')
        result = await mcp_server.add_image(image_base64=encode_image_to_base64(img))
        
        assert result["metadata"]["index"] == initial_count
    
    @pytest.mark.asyncio
    async def test_add_image_increments_total(self, mcp_server):
        """Test that total_embeddings increments correctly."""
        img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8), mode='L')
        
        result1 = await mcp_server.add_image(image_base64=encode_image_to_base64(img))
        total1 = result1["total_embeddings"]
        
        result2 = await mcp_server.add_image(image_base64=encode_image_to_base64(img))
        total2 = result2["total_embeddings"]
        
        assert total2 == total1 + 1
    
    @pytest.mark.asyncio
    async def test_add_image_path_is_valid(self, mcp_server):
        """Test that returned image path is valid and exists."""
        img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8), mode='L')
        
        result = await mcp_server.add_image(image_base64=encode_image_to_base64(img))
        
        image_path = Path(result["image_path"])
        assert image_path.exists()
        assert image_path.is_file()


class TestSearchByLabelExtended:
    """Extended tests for search_by_label functionality."""
    
    @pytest.mark.asyncio
    async def test_search_by_label_includes_human_readable(self, server_with_data):
        """Test that search_by_label always includes human_readable_label."""
        result = await server_with_data.search_by_label(label=0)
        
        assert "human_readable_label" in result
        assert isinstance(result["human_readable_label"], str)
        assert len(result["human_readable_label"]) > 0
    
    @pytest.mark.asyncio
    async def test_search_by_label_metadatas_is_list(self, server_with_data):
        """Test that metadatas is always a list."""
        result = await server_with_data.search_by_label(label=0)
        
        assert "metadatas" in result
        assert isinstance(result["metadatas"], list)
    
    @pytest.mark.asyncio
    async def test_search_by_label_ids_is_list(self, server_with_data):
        """Test that ids is always a list."""
        result = await server_with_data.search_by_label(label=0)
        
        assert "ids" in result
        assert isinstance(result["ids"], list)


class TestListAvailableLabelsExtended:
    """Extended tests for list_available_labels."""
    
    @pytest.mark.asyncio
    async def test_list_labels_returns_all_organ_labels(self, mcp_server):
        """Test that all 11 organ labels are returned."""
        result = await mcp_server.list_available_labels()
        
        assert "labels" in result
        assert "count" in result
        assert result["count"] == 11  # OrganSMNIST has 11 labels
    
    @pytest.mark.asyncio
    async def test_list_labels_has_all_keys(self, mcp_server):
        """Test that all label keys 0-10 are present."""
        result = await mcp_server.list_available_labels()
        
        labels_dict = result["labels"]
        for i in range(11):
            assert i in labels_dict or str(i) in labels_dict
    
    @pytest.mark.asyncio
    async def test_list_labels_values_are_strings(self, mcp_server):
        """Test that all label values are strings."""
        result = await mcp_server.list_available_labels()
        
        labels_dict = result["labels"]
        for value in labels_dict.values():
            assert isinstance(value, str)
            assert len(value) > 0


class TestToolCallResultFormat:
    """Test the format of tool call results."""
    
    @pytest.mark.asyncio
    async def test_successful_tool_call_format(self, mcp_server):
        """Test that successful tool calls have correct format."""
        result = await mcp_server.handle_tool_call("get_statistics", {})
        
        assert "success" in result
        assert result["success"] is True
        assert "result" in result
        assert "error" not in result or result.get("error") is None
    
    @pytest.mark.asyncio
    async def test_failed_tool_call_format(self, mcp_server):
        """Test that failed tool calls have correct format."""
        result = await mcp_server.handle_tool_call(
            "search_similar_images",
            {"image_base64": "not_valid_base64"}
        )
        
        assert "success" in result
        assert result["success"] is False
        assert "error" in result
        assert isinstance(result["error"], str)
    
    @pytest.mark.asyncio
    async def test_unknown_tool_format(self, mcp_server):
        """Test that unknown tool calls have correct error format."""
        result = await mcp_server.handle_tool_call("unknown_tool", {})
        
        assert "error" in result
        assert "available_tools" in result
        assert isinstance(result["available_tools"], list)
        assert len(result["available_tools"]) == 9


class TestClearStore:
    """Comprehensive tests for clear_store functionality."""
    
    @pytest.mark.asyncio
    async def test_clear_embeddings_only(self, server_with_data):
        """Test clearing embeddings while keeping images."""
        initial_images = server_with_data.image_store.count()
        initial_embeddings = server_with_data.rag_store.count()
        
        assert initial_embeddings > 0
        assert initial_images > 0
        
        result = await server_with_data.clear_store(
            clear_embeddings=True,
            clear_images=False
        )
        
        assert result["success"] is True
        assert result["embeddings_cleared"] == initial_embeddings
        assert result["images_deleted"] == 0
        assert result["embeddings_remaining"] == 0
        assert result["images_remaining"] == initial_images
        assert server_with_data.rag_store.count() == 0
        assert server_with_data.image_store.count() == initial_images
    
    @pytest.mark.asyncio
    async def test_clear_images_only(self, server_with_data):
        """Test clearing images while keeping embeddings."""
        initial_images = server_with_data.image_store.count()
        initial_embeddings = server_with_data.rag_store.count()
        
        assert initial_embeddings > 0
        assert initial_images > 0
        
        result = await server_with_data.clear_store(
            clear_embeddings=False,
            clear_images=True
        )
        
        assert result["success"] is True
        assert result["embeddings_cleared"] == 0
        assert result["images_deleted"] == initial_images
        assert result["embeddings_remaining"] == initial_embeddings
        assert result["images_remaining"] == 0
        assert server_with_data.rag_store.count() == initial_embeddings
        assert server_with_data.image_store.count() == 0
    
    @pytest.mark.asyncio
    async def test_clear_both_embeddings_and_images(self, server_with_data):
        """Test clearing both embeddings and images."""
        initial_images = server_with_data.image_store.count()
        initial_embeddings = server_with_data.rag_store.count()
        
        result = await server_with_data.clear_store(
            clear_embeddings=True,
            clear_images=True
        )
        
        assert result["success"] is True
        assert result["embeddings_cleared"] == initial_embeddings
        assert result["images_deleted"] == initial_images
        assert result["embeddings_remaining"] == 0
        assert result["images_remaining"] == 0
        assert server_with_data.rag_store.count() == 0
        assert server_with_data.image_store.count() == 0
    
    @pytest.mark.asyncio
    async def test_clear_empty_store(self, mcp_server):
        """Test clearing an already empty store."""
        result = await mcp_server.clear_store(
            clear_embeddings=True,
            clear_images=True
        )
        
        assert result["success"] is True
        assert result["embeddings_cleared"] == 0
        assert result["images_deleted"] == 0
    
    @pytest.mark.asyncio
    async def test_clear_neither_flag(self, server_with_data):
        """Test clear_store with both flags False."""
        initial_images = server_with_data.image_store.count()
        initial_embeddings = server_with_data.rag_store.count()
        
        result = await server_with_data.clear_store(
            clear_embeddings=False,
            clear_images=False
        )
        
        assert result["success"] is True
        assert result["embeddings_cleared"] == 0
        assert result["images_deleted"] == 0
        assert server_with_data.rag_store.count() == initial_embeddings
        assert server_with_data.image_store.count() == initial_images


class TestReindexFromImages:
    """Comprehensive tests for reindex_from_images functionality."""
    
    @pytest.mark.asyncio
    async def test_reindex_with_existing_images(self, server_with_data):
        """Test reindexing when images exist on disk."""
        # Clear embeddings but keep images
        await server_with_data.clear_store(clear_embeddings=True, clear_images=False)
        
        assert server_with_data.rag_store.count() == 0
        initial_images = server_with_data.image_store.count()
        assert initial_images > 0
        
        result = await server_with_data.reindex_from_images()
        
        assert result["success"] is True
        assert result["images_processed"] == initial_images
        assert result["images_skipped"] == 0
        assert result["embeddings_before"] == 0
        assert result["total_embeddings"] == initial_images
        assert result["cleared_before_reindex"] is False
    
    @pytest.mark.asyncio
    async def test_reindex_with_max_images(self, server_with_data):
        """Test reindexing with max_images limit."""
        await server_with_data.clear_store(clear_embeddings=True, clear_images=False)
        
        result = await server_with_data.reindex_from_images(max_images=3)
        
        assert result["success"] is True
        assert result["images_processed"] <= 3
        assert result["total_embeddings"] <= 3
    
    @pytest.mark.asyncio
    async def test_reindex_with_clear_existing(self, server_with_data):
        """Test reindexing with clear_existing=True."""
        initial_embeddings = server_with_data.rag_store.count()
        
        result = await server_with_data.reindex_from_images(clear_existing=True)
        
        assert result["success"] is True
        assert result["embeddings_before"] == initial_embeddings
        assert result["cleared_before_reindex"] is True
    
    @pytest.mark.asyncio
    async def test_reindex_no_images_error(self, tmp_path):
        """Test reindexing when no images exist."""
        # Create a fresh server with empty image store using temporary directories
        test_server = VisionRAGMCPServer(
            collection_name="test_reindex_empty",
            persist_directory=str(tmp_path / "chroma_db"),
            image_store_dir=str(tmp_path / "image_store")
        )
        
        try:
            # Clear everything to ensure empty state
            await test_server.clear_store(clear_embeddings=True, clear_images=True)
            
            result = await test_server.reindex_from_images()
            
            assert result["success"] is False
            assert "error" in result
            assert "No images found" in result["error"]
            assert "image_store_directory" in result
        finally:
            # Cleanup - clear both rag_store and image_store
            test_server.rag_store.clear()
            test_server.image_store.clear()
    
    @pytest.mark.asyncio
    async def test_reindex_increments_count(self, server_with_data):
        """Test that reindexing adds to existing embeddings when not clearing."""
        
        # Add more images
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8), mode='L')
            await server_with_data.add_image(encode_image_to_base64(img))
        
        # Clear embeddings but keep images
        await server_with_data.clear_store(clear_embeddings=True, clear_images=False)
        total_images = server_with_data.image_store.count()
        
        result = await server_with_data.reindex_from_images(clear_existing=False)
        
        assert result["success"] is True
        assert result["total_embeddings"] == total_images


class TestPreloadDataset:
    """Comprehensive tests for preload_dataset functionality."""
    
    @pytest.mark.asyncio
    async def test_preload_invalid_dataset(self, mcp_server):
        """Test preloading with invalid dataset name."""
        result = await mcp_server.preload_dataset(
            dataset_name="InvalidDataset",
            split="train",
            max_images=10
        )
        
        assert result["success"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_preload_with_max_images(self, mcp_server):
        """Test preloading with max_images limit."""
        initial_count = mcp_server.rag_store.count()
        
        result = await mcp_server.preload_dataset(
            dataset_name="PneumoniaMNIST",
            split="train",
            max_images=5,
            size=28
        )
        
        if result["success"]:
            assert result["images_loaded"] <= 5
            assert result["total_embeddings"] == initial_count + result["images_loaded"]
    
    @pytest.mark.asyncio
    async def test_preload_multiple_datasets(self, mcp_server):
        """Test preloading multiple datasets sequentially."""
        result1 = await mcp_server.preload_dataset(
            dataset_name="PneumoniaMNIST",
            split="train",
            max_images=3,
            size=28
        )
        
        if result1["success"]:
            count_after_first = result1["total_embeddings"]
            
            result2 = await mcp_server.preload_dataset(
                dataset_name="BreastMNIST",
                split="train",
                max_images=3,
                size=28
            )
            
            if result2["success"]:
                assert result2["total_embeddings"] >= count_after_first
    
    @pytest.mark.asyncio
    async def test_preload_includes_metadata(self, mcp_server):
        """Test that preloaded images have proper metadata."""
        await mcp_server.clear_store(clear_embeddings=True, clear_images=True)
        
        result = await mcp_server.preload_dataset(
            dataset_name="PneumoniaMNIST",
            split="train",
            max_images=2,
            size=28
        )
        
        if result["success"] and result["images_loaded"] > 0:
            # Search to verify metadata
            stats = await mcp_server.get_statistics()
            assert stats["total_embeddings"] > 0


class TestSearchSimilarImagesExtended:
    """Extended tests for search_similar_images."""
    
    @pytest.mark.asyncio
    async def test_search_returns_proper_structure(self, server_with_data):
        """Test that search returns proper result structure."""
        query_img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8), mode='L')
        
        result = await server_with_data.search_similar_images(
            image_base64=encode_image_to_base64(query_img),
            n_results=2
        )
        
        assert "results" in result
        assert "count" in result
        assert "query_info" in result
    
    @pytest.mark.asyncio
    async def test_search_with_varying_n_results(self, server_with_data):
        """Test search with different n_results values."""
        query_img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8), mode='L')
        
        result1 = await server_with_data.search_similar_images(
            image_base64=encode_image_to_base64(query_img),
            n_results=1
        )
        
        result3 = await server_with_data.search_similar_images(
            image_base64=encode_image_to_base64(query_img),
            n_results=3
        )
        
        assert result1["count"] <= 1
        assert result3["count"] <= 3


class TestSearchByLabelWithImages:
    """Test search_by_label with return_images parameter."""
    
    @pytest.mark.asyncio
    async def test_search_by_label_with_images(self, server_with_data):
        """Test search_by_label with return_images=True."""
        result = await server_with_data.search_by_label(
            label=0,
            n_results=2,
            return_images=True
        )
        
        if result["count"] > 0:
            assert "images_base64" in result
            assert isinstance(result["images_base64"], list)
    
    @pytest.mark.asyncio
    async def test_search_by_label_without_images(self, server_with_data):
        """Test search_by_label with return_images=False."""
        result = await server_with_data.search_by_label(
            label=0,
            n_results=2,
            return_images=False
        )
        
        assert "images_base64" not in result


class TestGetStatisticsExtended:
    """Extended tests for get_statistics."""
    
    @pytest.mark.asyncio
    async def test_statistics_has_all_required_fields(self, mcp_server):
        """Test that statistics includes all required fields."""
        stats = await mcp_server.get_statistics()
        
        required_fields = [
            "total_embeddings",
            "total_images",
            "collection_name",
            "persist_directory",
            "image_store_directory",
            "current_working_directory",
            "encoder_model",
            "embedding_dimension"
        ]
        
        for field in required_fields:
            assert field in stats, f"Missing field: {field}"
    
    @pytest.mark.asyncio
    async def test_statistics_counts_are_non_negative(self, mcp_server):
        """Test that all counts in statistics are non-negative."""
        stats = await mcp_server.get_statistics()
        
        assert stats["total_embeddings"] >= 0
        assert stats["total_images"] >= 0
        assert stats["embedding_dimension"] > 0
    
    @pytest.mark.asyncio
    async def test_statistics_paths_are_absolute(self, mcp_server):
        """Test that directory paths in statistics are absolute."""
        stats = await mcp_server.get_statistics()
        
        assert Path(stats["persist_directory_absolute"]).is_absolute()
        assert Path(stats["image_store_directory_absolute"]).is_absolute()
        assert Path(stats["current_working_directory"]).is_absolute()


class TestListAvailableDatasets:
    """Tests for list_available_datasets functionality."""
    
    @pytest.mark.asyncio
    async def test_list_datasets_returns_dict(self, mcp_server):
        """Test that list_available_datasets returns proper structure."""
        result = await mcp_server.list_available_datasets()
        
        assert "datasets" in result
        assert "count" in result
        assert isinstance(result["datasets"], dict)
        assert isinstance(result["count"], int)
        assert result["count"] > 0
    
    @pytest.mark.asyncio
    async def test_list_datasets_includes_common_datasets(self, mcp_server):
        """Test that common MedMNIST datasets are included."""
        result = await mcp_server.list_available_datasets()
        
        datasets = result["datasets"]
        common_datasets = ["PathMNIST", "ChestMNIST", "PneumoniaMNIST", "OrganSMNIST"]
        
        for ds_name in common_datasets:
            assert ds_name in datasets, f"Missing dataset: {ds_name}"
    
    @pytest.mark.asyncio
    async def test_list_datasets_has_proper_info(self, mcp_server):
        """Test that each dataset has required information."""
        result = await mcp_server.list_available_datasets()
        
        for ds_name, ds_info in result["datasets"].items():
            assert "description" in ds_info
            assert "n_classes" in ds_info
            assert "image_size" in ds_info
            assert "channels" in ds_info
            
            assert isinstance(ds_info["n_classes"], int)
            assert ds_info["n_classes"] > 0


class TestErrorHandling:
    """Test error handling across all functions."""
    
    @pytest.mark.asyncio
    async def test_search_with_corrupted_base64(self, mcp_server):
        """Test search with corrupted base64 string raises error."""
        with pytest.raises(ValueError):
            await mcp_server.search_similar_images(
                image_base64="corrupted_data_!!!",
                n_results=5
            )
    
    @pytest.mark.asyncio
    async def test_add_image_with_invalid_base64(self, mcp_server):
        """Test add_image with invalid base64 raises error."""
        with pytest.raises(ValueError):
            await mcp_server.add_image(
                image_base64="not_a_valid_image",
                metadata={"test": True}
            )
    
    @pytest.mark.asyncio
    async def test_search_by_label_with_negative_label(self, server_with_data):
        """Test search_by_label with negative label value."""
        result = await server_with_data.search_by_label(label=-1)
        
        # Should handle gracefully, return empty results
        assert "ids" in result
        assert "metadatas" in result
        assert "count" in result
        assert result["count"] == 0
    
    @pytest.mark.asyncio
    async def test_preload_with_invalid_split(self, mcp_server):
        """Test preload with invalid split name."""
        result = await mcp_server.preload_dataset(
            dataset_name="PneumoniaMNIST",
            split="invalid_split",
            max_images=5
        )
        
        assert result["success"] is False
        assert "error" in result
