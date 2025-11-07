"""Tests for MCP server module."""

import pytest
import asyncio
import numpy as np
from PIL import Image

from vision_rag.mcp_server import VisionRAGMCPServer
from vision_rag.utils import encode_image_to_base64


@pytest.fixture
def mcp_server():
    """Create an MCP server instance for testing."""
    server = VisionRAGMCPServer(
        collection_name="test_mcp_collection",
        persist_directory="./test_mcp_chroma_db",
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
        assert len(mcp_server.tools) == 5
        
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
        assert "total_embeddings" in result
        assert result["metadata"]["label"] == 5
        assert result["metadata"]["source"] == "test"
        assert result["total_embeddings"] == initial_count + 1
    
    @pytest.mark.asyncio
    async def test_add_image_without_metadata(self, mcp_server, sample_image_base64):
        """Test adding image without metadata."""
        result = await mcp_server.add_image(
            image_base64=sample_image_base64
        )
        
        assert "id" in result
        assert "metadata" in result
        assert "index" in result["metadata"]
    
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


class TestGetStatistics:
    """Tests for the get_statistics tool."""
    
    @pytest.mark.asyncio
    async def test_get_statistics_empty(self, mcp_server):
        """Test getting statistics from empty store."""
        result = await mcp_server.get_statistics()
        
        assert "total_embeddings" in result
        assert "collection_name" in result
        assert "persist_directory" in result
        assert "encoder_model" in result
        assert "embedding_dimension" in result
        
        assert isinstance(result["total_embeddings"], int)
        assert isinstance(result["embedding_dimension"], int)
    
    @pytest.mark.asyncio
    async def test_get_statistics_with_data(self, server_with_data):
        """Test getting statistics with data in store."""
        result = await server_with_data.get_statistics()
        
        assert result["total_embeddings"] == 5
        assert result["collection_name"] == "test_mcp_collection"
        assert result["embedding_dimension"] > 0


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
        assert len(definitions) == 5
        
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
        assert search_def["parameters"]["image_base64"]["required"] is True
        assert search_def["parameters"]["n_results"]["required"] is False


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
