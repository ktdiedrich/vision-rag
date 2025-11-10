"""
Example MCP client demonstrating how to interact with the Vision RAG MCP server.

This example shows how to:
1. Connect to the MCP server
2. List available tools
3. Get statistics
4. List available labels
5. Search by label
6. Add an image
7. Search for similar images

Usage:
    python examples/mcp_client_example.py
"""

import asyncio
import sys
import json
import base64
import io
from pathlib import Path
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Import MedMNIST
try:
    import medmnist
except ImportError:
    print("Error: medmnist package not found. Install with: pip install medmnist")
    sys.exit(1)


async def main():
    """Run MCP client example."""
    print("=" * 60)
    print("Vision RAG MCP Client Example")
    print("=" * 60)
    print()
    
    try:
        # Configure server parameters
        server_params = StdioServerParameters(
            command="python",
            args=[
                str(Path(__file__).parent.parent / "scripts" / "run_service.py"),
                "--mode",
                "mcp"
            ],
            env=None
        )
        
        print("📡 Connecting to MCP server...")
        
        # Connect to server
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                print("✓ Connected to MCP server\n")
                
                # 1. List available tools
                print("1️⃣  Listing available tools...")
                tools_response = await session.list_tools()
                print(f"   Found {len(tools_response.tools)} tools:")
                for tool in tools_response.tools:
                    print(f"   - {tool.name}: {tool.description}")
                print()
                
                # 2. Get statistics
                print("2️⃣  Getting RAG store statistics...")
                stats_result = await session.call_tool("get_statistics", {})
                stats = json.loads(stats_result.content[0].text)
                if stats.get("success"):
                    stats_data = stats["result"]
                    print(f"   Total embeddings: {stats_data['total_embeddings']}")
                    print(f"   Total images: {stats_data['total_images']}")
                    print(f"   Encoder model: {stats_data['encoder_model']}")
                    print(f"   Embedding dimension: {stats_data['embedding_dimension']}")
                print()
                
                # 3. List available labels
                print("3️⃣  Listing available organ labels...")
                labels_result = await session.call_tool("list_available_labels", {})
                labels = json.loads(labels_result.content[0].text)
                if labels.get("success"):
                    labels_data = labels["result"]
                    print(f"   Found {labels_data['count']} labels:")
                    for label_id, label_name in sorted(labels_data["labels"].items(), key=lambda x: int(x[0])):
                        print(f"   - {label_id}: {label_name}")
                print()
                
                # 4. Search by label (without images)
                print("4️⃣  Searching for heart images (label=3)...")
                search_result = await session.call_tool("search_by_label", {
                    "label": 3,
                    "n_results": 3,
                    "return_images": False
                })
                
                if not search_result.content or not search_result.content[0].text:
                    print(f"   ⚠️  Empty response from server")
                else:
                    search_data = json.loads(search_result.content[0].text)
                    if search_data.get("success"):
                        result = search_data["result"]
                        print(f"   Label: {result['label']} ({result['human_readable_label']})")
                        print(f"   Found: {result['count']} images")
                        if result['count'] > 0:
                            print(f"   Sample IDs: {result['ids'][:3]}")
                        else:
                            print(f"   ℹ️  No images found with label {result['label']}")
                    else:
                        print(f"   ❌ Error: {search_data.get('error')}")
                print()
                
                # 5. Load real medical images from OrganCMNIST
                print("5️⃣  Loading OrganCMNIST dataset (224x224)...")
                
                # Download and load OrganCMNIST
                DataClass = getattr(medmnist, 'OrganCMNIST')
                dataset = DataClass(split='train', download=True, size=224, root='./data')
                
                # Get a few sample images (heart images - label 3)
                print(f"   Dataset loaded: {len(dataset)} images")
                print(f"   Looking for heart images (label=3)...")
                
                # Find heart images
                heart_images = []
                for idx in range(min(100, len(dataset))):  # Check first 100 images
                    img, label = dataset[idx]
                    if label == 3:  # heart
                        heart_images.append((img, idx))
                        if len(heart_images) >= 3:
                            break
                
                if not heart_images:
                    print(f"   ⚠️  No heart images found in first 100 samples")
                    print(f"   Using first available image instead")
                    img, label = dataset[0]
                    heart_images = [(img, 0)]
                    label_name = "other"
                else:
                    print(f"   Found {len(heart_images)} heart images")
                    label_name = "heart"
                
                # Add the first heart image
                img_array, img_idx = heart_images[0]
                
                # Convert numpy array to PIL Image
                # MedMNIST returns images as numpy arrays with shape (H, W, C) or (H, W)
                if isinstance(img_array, np.ndarray):
                    if img_array.ndim == 3 and img_array.shape[2] == 1:
                        # Remove channel dimension if grayscale
                        img_array = img_array.squeeze(axis=2)
                    test_image = Image.fromarray(img_array.astype(np.uint8), mode='L')
                else:
                    test_image = img_array
                
                print(f"   Adding image {img_idx} (label={label_name}, size={test_image.size})...")
                
                # Encode to base64
                buffer = io.BytesIO()
                test_image.save(buffer, format='PNG')
                test_image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                add_result = await session.call_tool("add_image", {
                    "image_base64": test_image_b64,
                    "metadata": {
                        "label": 3,  # heart
                        "source": "OrganCMNIST",
                        "description": f"OrganCMNIST image {img_idx}",
                        "dataset_index": img_idx
                    }
                })
                add_data = json.loads(add_result.content[0].text)
                if add_data.get("success"):
                    print(f"   ✓ Added image with ID: {add_data['result']['id']}")
                print()
                
                # 6. Now search by label (should find our added image)
                print("6️⃣  Searching for heart images again (label=3)...")
                search_result2 = await session.call_tool("search_by_label", {
                    "label": 3,
                    "n_results": 3,
                    "return_images": False
                })
                search_data2 = json.loads(search_result2.content[0].text)
                if search_data2.get("success"):
                    result = search_data2["result"]
                    print(f"   Found: {result['count']} images")
                    if result['count'] > 0:
                        print(f"   IDs: {result['ids']}")
                print()
                
                # 7. Search by label (with images)
                print("7️⃣  Searching for heart images with base64 data (label=3)...")
                search_with_images = await session.call_tool("search_by_label", {
                    "label": 3,
                    "n_results": 2,
                    "return_images": True
                })
                search_images_data = json.loads(search_with_images.content[0].text)
                if search_images_data.get("success"):
                    result = search_images_data["result"]
                    print(f"   Found: {result['count']} images")
                    if "images_base64" in result and result['count'] > 0:
                        first_image = result["images_base64"][0]
                        if first_image:
                            print(f"   First image base64 length: {len(first_image)} characters")
                            print(f"   Images can be decoded and displayed")
                print()
                
                # 8. Search for similar images using our test image
                print("8️⃣  Searching for similar images...")
                similar_result = await session.call_tool("search_similar_images", {
                    "image_base64": test_image_b64,
                    "n_results": 3
                })
                similar_data = json.loads(similar_result.content[0].text)
                if similar_data.get("success"):
                    result = similar_data["result"]
                    print(f"   Found {result['count']} similar images")
                    if result['count'] > 0:
                        for i, res in enumerate(result['results'][:3], 1):
                            print(f"   {i}. ID: {res['id']}, Distance: {res['distance']:.4f}")
                print()
                
                print("=" * 60)
                print("✅ MCP client example completed successfully!")
                print("=" * 60)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
