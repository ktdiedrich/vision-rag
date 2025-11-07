"""Model Context Protocol (MCP) server for vision RAG agent communication."""

import asyncio
from typing import Any, Dict, List, Optional

from PIL import Image

from .config import CLIP_MODEL_NAME, MEDMNIST_DATASET
from .encoder import CLIPImageEncoder
from .rag_store import ChromaRAGStore
from .search import ImageSearcher
from .data_loader import get_human_readable_label
from .utils import decode_base64_image


class VisionRAGMCPServer:
    """MCP server for vision RAG system enabling agent-to-agent communication."""
    
    def __init__(
        self,
        encoder_model: str = "clip-ViT-B-32",
        collection_name: str = "mcp_vision_rag",
        persist_directory: str = "./chroma_db_mcp",
    ):
        """
        Initialize MCP server.
        
        Args:
            collection_name: ChromaDB collection name
            persist_directory: Directory for persistent storage
        """
        self.encoder = CLIPImageEncoder(model_name=CLIP_MODEL_NAME)
        self.rag_store = ChromaRAGStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
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
    ) -> Dict[str, Any]:
        """
        Search for images by label.
        
        Args:
            label: Label to search for
            n_results: Optional limit on results
            
        Returns:
            Images with matching label
        """
        results = self.rag_store.search_by_label(label=label, n_results=n_results)
        
        return {
            "label": label,
            "human_readable_label": get_human_readable_label(label, dataset_name=MEDMNIST_DATASET),
            "ids": results["ids"],
            "metadatas": results["metadatas"],
            "count": len(results["ids"]),
        }
    
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
        
        # Encode image
        embedding = self.encoder.encode_image(image)
        
        # Generate ID
        current_count = self.rag_store.count()
        image_id = f"img_{current_count}"
        
        # Prepare metadata
        if metadata is None:
            metadata = {"index": current_count}
        
        # Add to store
        self.rag_store.add_embeddings(
            embeddings=embedding.reshape(1, -1),
            ids=[image_id],
            metadatas=[metadata],
        )
        
        return {
            "id": image_id,
            "metadata": metadata,
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
            "collection_name": self.rag_store.collection_name,
            "persist_directory": self.rag_store.persist_directory,
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
                        "required": True,
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of similar images to return (default: 5)",
                        "required": False,
                        "default": 5,
                    },
                },
            },
            {
                "name": "search_by_label",
                "description": "Search for medical images by organ label",
                "parameters": {
                    "label": {
                        "type": "integer",
                        "description": "Organ label (0-10: bladder, femur-left, femur-right, heart, kidney-left, kidney-right, liver, lung-left, lung-right, pancreas, spleen)",
                        "required": True,
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Optional limit on number of results",
                        "required": False,
                    },
                },
            },
            {
                "name": "add_image",
                "description": "Add a medical image to the RAG store",
                "parameters": {
                    "image_base64": {
                        "type": "string",
                        "description": "Base64 encoded image to add",
                        "required": True,
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata (e.g., label, patient_id)",
                        "required": False,
                    },
                },
            },
            {
                "name": "get_statistics",
                "description": "Get statistics about the vision RAG store",
                "parameters": {},
            },
            {
                "name": "list_available_labels",
                "description": "List all available organ labels with human-readable names",
                "parameters": {},
            },
        ]


async def main():
    """Run MCP server."""
    server = VisionRAGMCPServer()
    
    print("🤖 Vision RAG MCP Server Started")
    print(f"📊 Total embeddings: {server.rag_store.count()}")
    print(f"🔧 Available tools: {list(server.tools.keys())}")
    print("\n📋 Tool Definitions:")
    
    for tool in server.get_tool_definitions():
        print(f"  - {tool['name']}: {tool['description']}")
    
    print("\n✅ Server ready for agent communication")
    
    # Keep server running
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
