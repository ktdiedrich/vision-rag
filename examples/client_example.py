"""
Example client demonstrating how to interact with the Vision RAG service.

This shows how AI agents can communicate with the Vision RAG service.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import base64
import io
import httpx
from PIL import Image


async def test_api_service():
    """Test the FastAPI service endpoints."""
    base_url = "http://localhost:8001"
    
    async with httpx.AsyncClient() as client:
        # 1. Health check
        print("1️⃣  Checking service health...")
        response = await client.get(f"{base_url}/health")
        print(f"   Status: {response.json()}")
        
        # 2. Get statistics
        print("\n2️⃣  Getting statistics...")
        response = await client.get(f"{base_url}/stats")
        print(f"   Stats: {response.json()}")
        
        # 3. Add a test image
        print("\n3️⃣  Adding a test image...")
        # Create a simple test image
        test_image = Image.new('L', (28, 28), color=128)
        buffer = io.BytesIO()
        test_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        response = await client.post(
            f"{base_url}/add",
            json={
                "image_base64": image_base64,
                "metadata": {"label": 6, "description": "test liver image"}
            }
        )
        result = response.json()
        print(f"   Added: {result}")
        
        # 4. Search for similar images
        print("\n4️⃣  Searching for similar images...")
        response = await client.post(
            f"{base_url}/search",
            json={
                "image_base64": image_base64,
                "n_results": 3
            }
        )
        search_results = response.json()
        print(f"   Found {search_results['count']} results")
        for result in search_results['results']:
            print(f"   - {result['id']}: distance={result['distance']:.3f}, "
                  f"label={result.get('human_readable_label', 'N/A')}")
        
        # 5. Search by label
        print("\n5️⃣  Searching by label (liver = 6)...")
        response = await client.post(
            f"{base_url}/search/label",
            json={"label": 6, "n_results": 5}
        )
        label_results = response.json()
        print(f"   Found {label_results['count']} images with label 'liver'")


async def test_mcp_server():
    """Test the MCP server for agent communication."""
    from vision_rag.mcp_server import VisionRAGMCPServer
    
    print("\n" + "=" * 60)
    print("🤖 Testing MCP Agent Server")
    print("=" * 60)
    
    # Initialize MCP server
    server = VisionRAGMCPServer()
    
    # 1. Get tool definitions
    print("\n1️⃣  Available tools:")
    tools = server.get_tool_definitions()
    for tool in tools:
        print(f"   - {tool['name']}: {tool['description']}")
    
    # 2. Test get statistics
    print("\n2️⃣  Getting statistics...")
    result = await server.handle_tool_call("get_statistics", {})
    if result["success"]:
        print(f"   Stats: {result['result']}")
    
    # 3. Test list available labels
    print("\n3️⃣  Listing available labels...")
    result = await server.handle_tool_call("list_available_labels", {})
    if result["success"]:
        labels = result['result']['labels']
        print(f"   Found {len(labels)} labels:")
        for label_id, label_name in labels.items():
            print(f"   - {label_id}: {label_name}")
    
    # 4. Add an image via MCP
    print("\n4️⃣  Adding image via MCP...")
    test_image = Image.new('L', (28, 28), color=200)
    buffer = io.BytesIO()
    test_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    result = await server.handle_tool_call(
        "add_image",
        {
            "image_base64": image_base64,
            "metadata": {"label": 3, "source": "mcp_client"}
        }
    )
    if result["success"]:
        print(f"   Added: {result['result']}")
    
    # 5. Search by label via MCP
    print("\n5️⃣  Searching by label via MCP...")
    result = await server.handle_tool_call(
        "search_by_label",
        {"label": 3, "n_results": 3}
    )
    if result["success"]:
        search_result = result['result']
        print(f"   Found {search_result['count']} images for "
              f"{search_result['human_readable_label']}")


async def main():
    """Run client tests."""
    print("=" * 60)
    print("🔬 Vision RAG Service Client Demo")
    print("=" * 60)
    
    # Test MCP server (doesn't require service to be running)
    await test_mcp_server()
    
    # Test API service (requires service to be running)
    print("=" * 60)
    print("🌐 Testing FastAPI Service")
    print("=" * 60)
    print("\n⚠️  Make sure the service is running:")
    print("   python scripts/run_service.py --mode api\n")
    
    try:
        await test_api_service()
    except httpx.ConnectError:
        print("❌ Could not connect to API service")
        print("   Start the service with: python scripts/run_service.py --mode api")
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())