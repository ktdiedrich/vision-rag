#!/usr/bin/env python3
"""
Demo script showing how to generate t-SNE visualizations from the RAG store.

This example demonstrates:
1. Using the MCP server to generate t-SNE plots
2. Using the FastAPI service endpoint
3. Different visualization methods (t-SNE, PCA, UMAP)
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision_rag.mcp_server import VisionRAGMCPServer


async def demo_mcp_tsne():
    """Demonstrate generating t-SNE plots using the MCP server."""
    print("=" * 80)
    print("MCP Server t-SNE Visualization Demo")
    print("=" * 80)
    print()
    
    # Initialize MCP server
    print("📊 Initializing Vision RAG MCP Server...")
    server = VisionRAGMCPServer(
        collection_name="mcp_tsne_demo",
        persist_directory="./chroma_db_mcp",
        image_store_dir="./image_store_mcp",
    )
    
    # Check if we have embeddings
    total_embeddings = server.rag_store.count()
    print(f"📈 Total embeddings in store: {total_embeddings}")
    
    if total_embeddings == 0:
        print("\n⚠️  No embeddings found in the RAG store.")
        print("   Please run a demo that adds images first, such as:")
        print("   - demonstrations/mcp_preload_demo.py")
        print("   - demonstrations/simple_visualization_demo.py")
        return
    
    print(f"\n✅ Found {total_embeddings} embeddings to visualize")
    
    # Generate t-SNE plot
    print("\n🎨 Generating t-SNE visualization...")
    result = await server.generate_tsne_plot(
        output_filename="mcp_tsne_visualization.png",
        method="tsne",
        title="MCP RAG Store - t-SNE Embedding Visualization",
    )
    
    if result["success"]:
        print(f"✅ {result['message']}")
        print(f"📁 Plot saved to: {result['output_path']}")
        print(f"   Total embeddings: {result['total_embeddings']}")
        print(f"   Unique labels: {result['unique_labels']}")
        print(f"   Method: {result['method'].upper()}")
    else:
        print(f"❌ Error: {result['error']}")
    
    # Generate PCA plot
    print("\n🎨 Generating PCA visualization...")
    result = await server.generate_tsne_plot(
        output_filename="mcp_pca_visualization.png",
        method="pca",
        title="MCP RAG Store - PCA Embedding Visualization",
    )
    
    if result["success"]:
        print(f"✅ {result['message']}")
        print(f"📁 Plot saved to: {result['output_path']}")
    else:
        print(f"❌ Error: {result['error']}")
    
    print("\n🎉 Demo complete!")


def demo_api_usage():
    """Show example API usage for the FastAPI endpoint."""
    print("\n" + "=" * 80)
    print("FastAPI Service t-SNE Endpoint Usage")
    print("=" * 80)
    print()
    
    endpoint_url = "http://localhost:8001/visualize/tsne"
    
    print("To use the FastAPI endpoint, start the service:")
    print("  uv run python -m vision_rag.service")
    print()
    print("Then use curl or any HTTP client:")
    print()
    print("# Generate t-SNE plot:")
    print(f'  curl -X POST {endpoint_url} \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"output_filename": "api_tsne.png", "method": "tsne", "title": "API t-SNE Visualization"}\'')
    print()
    print("# Generate PCA plot:")
    print(f'  curl -X POST {endpoint_url} \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"output_filename": "api_pca.png", "method": "pca"}\'')
    print()
    print("# Generate UMAP plot:")
    print(f'  curl -X POST {endpoint_url} \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"output_filename": "api_umap.png", "method": "umap"}\'')
    print()
    print("Or use Python requests:")
    print()
    print("  import requests")
    print(f'  response = requests.post("{endpoint_url}", json={{')
    print('      "output_filename": "my_tsne.png",')
    print('      "method": "tsne",')
    print('      "title": "My Custom Visualization"')
    print("  })")
    print("  print(response.json())")
    print()


async def main():
    """Run the demo."""
    # Demo MCP server functionality
    await demo_mcp_tsne()
    
    # Show API usage examples
    demo_api_usage()


if __name__ == "__main__":
    asyncio.run(main())
