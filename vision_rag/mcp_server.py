"""Model Context Protocol (MCP) server for vision RAG agent communication."""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .config import CLIP_MODEL_NAME, MEDMNIST_DATASET, IMAGE_SIZE
from .encoder import CLIPImageEncoder
from .rag_store import ChromaRAGStore
from .search import ImageSearcher
from .data_loader import get_human_readable_label
from .utils import decode_base64_image, encode_image_to_base64
from .image_store import ImageFileStore
from PIL import Image


class VisionRAGMCPServer:
    """MCP server for vision RAG system enabling agent-to-agent communication."""
    
    def __init__(
        self,
        encoder_model: str = CLIP_MODEL_NAME,
        collection_name: str = "mcp_vision_rag",
        persist_directory: str = "./chroma_db_mcp",
        image_store_dir: str = "./image_store_mcp",
        image_size: Optional[int] = IMAGE_SIZE,
    ):
        """
        Initialize MCP server.
        
        Args:
            encoder_model: CLIP model name
            collection_name: ChromaDB collection name
            persist_directory: Directory for persistent storage
            image_store_dir: Directory for image file storage
            image_size: Target size for storing images (width and height).
                       If provided, images will be resized to (image_size, image_size).
                       If None, images are stored at original size.
        """
        self.encoder = CLIPImageEncoder(model_name=encoder_model)
        self.rag_store = ChromaRAGStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        self.image_store = ImageFileStore(storage_dir=image_store_dir, image_size=image_size)
        self.searcher = ImageSearcher(encoder=self.encoder, rag_store=self.rag_store)
        
        # MCP protocol handlers
        self.tools = {
            "search_similar_images": self.search_similar_images,
            "search_by_label": self.search_by_label,
            "add_image": self.add_image,
            "get_statistics": self.get_statistics,
            "list_available_labels": self.list_available_labels,
        }
    
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming tool calls from other agents.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            return {
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self.tools.keys()),
            }
        
        try:
            result = await self.tools[tool_name](**arguments)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def search_similar_images(
        self,
        image_base64: str,
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Search for similar images.
        
        Args:
            image_base64: Base64 encoded image
            n_results: Number of results to return
            
        Returns:
            Search results with metadata
        """
        # Decode image
        image = decode_base64_image(image_base64)
        
        # Perform search
        results = self.searcher.search(image, n_results=n_results)
        
        # Enrich results with human-readable labels
        enriched_results = []
        for result_id, distance, metadata in zip(
            results["ids"], results["distances"], results["metadatas"]
        ):
            result_dict = {
                "id": result_id,
                "distance": float(distance),
                "metadata": metadata,
            }
            
            # Add human-readable label if available
            if "label" in metadata:
                result_dict["human_readable_label"] = get_human_readable_label(metadata["label"], dataset_name=MEDMNIST_DATASET)
            
            enriched_results.append(result_dict)
        
        return {
            "query_info": {
                "image_size": list(image.size),
                "image_mode": image.mode,
            },
            "results": enriched_results,
            "count": len(enriched_results),
        }
    
    async def search_by_label(
        self,
        label: int,
        n_results: Optional[int] = None,
        return_images: bool = False,
    ) -> Dict[str, Any]:
        """
        Search for images by label.
        
        Args:
            label: Label to search for
            n_results: Optional limit on results
            return_images: If True, return base64 encoded images
            
        Returns:
            Images with matching label, optionally including base64 encoded image data
        """
        results = self.rag_store.search_by_label(label=label, n_results=n_results)
        
        response = {
            "label": label,
            "human_readable_label": get_human_readable_label(label, dataset_name=MEDMNIST_DATASET),
            "ids": results["ids"],
            "metadatas": results["metadatas"],
            "count": len(results["ids"]),
        }
        
        # Optionally load and encode images
        if return_images:
            images = []
            for metadata in results["metadatas"]:
                if "image_path" in metadata:
                    image_path = Path(metadata["image_path"])
                    if image_path.exists():
                        try:
                            img = Image.open(image_path)
                            images.append(encode_image_to_base64(img))
                        except Exception:
                            # Handle corrupted or unopenable images
                            images.append(None)
                    else:
                        images.append(None)
                else:
                    images.append(None)
            response["images_base64"] = images
        
        return response
    
    async def add_image(
        self,
        image_base64: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add an image to the RAG store.
        
        Args:
            image_base64: Base64 encoded image
            metadata: Optional metadata
            
        Returns:
            Status and assigned ID
        """
        # Decode image
        image = decode_base64_image(image_base64)
        
        # Save image to disk
        image_path = self.image_store.save_image(image)
        
        # Encode image
        embedding = self.encoder.encode_image(image)
        
        # Generate ID
        current_count = self.rag_store.count()
        image_id = f"img_{current_count}"
        
        # Prepare metadata with image path
        if metadata is None:
            metadata = {}
        metadata["image_path"] = image_path
        metadata["index"] = current_count
        
        # Add to store
        self.rag_store.add_embeddings(
            embeddings=embedding.reshape(1, -1),
            ids=[image_id],
            metadatas=[metadata],
        )
        
        return {
            "id": image_id,
            "metadata": metadata,
            "image_path": image_path,
            "total_embeddings": self.rag_store.count(),
        }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get RAG store statistics.
        
        Returns:
            Statistics about the store
        """
        return {
            "total_embeddings": self.rag_store.count(),
            "total_images": self.image_store.count(),
            "collection_name": self.rag_store.collection_name,
            "persist_directory": self.rag_store.persist_directory,
            "image_store_directory": str(self.image_store.storage_dir),
            "encoder_model": CLIP_MODEL_NAME,
            "embedding_dimension": self.encoder.embedding_dimension,
        }
    
    async def list_available_labels(self) -> Dict[str, Any]:
        """
        List all available organ labels.
        
        Returns:
            Mapping of label IDs to human-readable names
        """
        from .data_loader import get_organmnist_label_names
        
        label_names = get_organmnist_label_names()
        
        return {
            "labels": label_names,
            "count": len(label_names),
        }
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get MCP tool definitions for agent discovery.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "name": "search_similar_images",
                "description": "Search for visually similar medical images using CLIP embeddings",
                "parameters": {
                    "image_base64": {
                        "type": "string",
                        "description": "Base64 encoded image to search for",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of similar images to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["image_base64"],
            },
            {
                "name": "search_by_label",
                "description": "Search for medical images by organ label",
                "parameters": {
                    "label": {
                        "type": "integer",
                        "description": "Organ label (0-10: bladder, femur-left, femur-right, heart, kidney-left, kidney-right, liver, lung-left, lung-right, pancreas, spleen)",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Optional limit on number of results",
                    },
                    "return_images": {
                        "type": "boolean",
                        "description": "If True, return base64 encoded images (default: False)",
                        "default": False,
                    },
                },
                "required": ["label"],
            },
            {
                "name": "add_image",
                "description": "Add a medical image to the RAG store",
                "parameters": {
                    "image_base64": {
                        "type": "string",
                        "description": "Base64 encoded image to add",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata (e.g., label, patient_id)",
                    },
                },
                "required": ["image_base64"],
            },
            {
                "name": "get_statistics",
                "description": "Get statistics about the vision RAG store",
                "parameters": {},
                "required": [],
            },
            {
                "name": "list_available_labels",
                "description": "List all available organ labels with human-readable names",
                "parameters": {},
                "required": [],
            },
        ]


async def main():
    """Run MCP server using stdio transport."""
    # Create the MCP server instance
    mcp_server = Server("vision-rag")
    
    # Initialize Vision RAG components
    vision_rag = VisionRAGMCPServer()
    
    print("🤖 Vision RAG MCP Server Starting...", file=sys.stderr)
    print(f"📊 Total embeddings: {vision_rag.rag_store.count()}", file=sys.stderr)
    print(f"🔧 Available tools: {list(vision_rag.tools.keys())}", file=sys.stderr)
    
    # Register tools with MCP server
    @mcp_server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available tools."""
        tools = []
        for tool_def in vision_rag.get_tool_definitions():
            # Convert our tool definitions to MCP Tool format with proper JSON Schema
            input_schema = {
                "type": "object",
                "properties": tool_def["parameters"],
            }
            # Add required field if it exists
            if "required" in tool_def and tool_def["required"]:
                input_schema["required"] = tool_def["required"]
            
            tools.append(Tool(
                name=tool_def["name"],
                description=tool_def["description"],
                inputSchema=input_schema
            ))
        return tools
    
    @mcp_server.call_tool()
    async def call_tool(name: str, arguments: dict) -> List[TextContent]:
        """Execute a tool by name."""
        try:
            print(f"🔧 Calling tool: {name} with arguments: {arguments}", file=sys.stderr)
            result = await vision_rag.handle_tool_call(name, arguments)
            
            result_json = json.dumps(result, indent=2)
            print(f"✅ Tool {name} completed successfully", file=sys.stderr)
            return [TextContent(
                type="text",
                text=result_json
            )]
        except Exception as e:
            print(f"❌ Error in tool {name}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
    
    print("✅ MCP Server ready for agent communication", file=sys.stderr)
    
    # Run the server with stdio transport
    try:
        async with stdio_server() as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options()
            )
    except Exception as e:
        print(f"❌ MCP Server error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise

if __name__ == "__main__":
    asyncio.run(main())
