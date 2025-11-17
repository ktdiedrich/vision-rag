"""
Demonstration of dataset preloading functionality.

This script shows how to use the preload endpoints to load MedMNIST datasets
into the Vision RAG store.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision_rag.mcp_server import VisionRAGMCPServer


async def demo_list_datasets():
    """Demo: List all available datasets."""
    print("\n" + "="*60)
    print("DEMO 1: List Available Datasets")
    print("="*60)
    
    server = VisionRAGMCPServer(
        collection_name="preload_demo",
        persist_directory="./chroma_db_preload_demo",
        image_store_dir="./image_store_preload_demo",
    )
    
    result = await server.list_available_datasets()
    if not result:
        print("❌ Error: No response from server")
        return
    
    print(f"\nFound {result['count']} available datasets:\n")
    for name, info in result['datasets'].items():
        print(f"  • {name}")
        print(f"    Description: {info['description']}")
        print(f"    Classes: {info['n_classes']}")
        print(f"    Image size: {info['image_size']}x{info['image_size']}")
        print(f"    Channels: {info['channels']}")
        print()


async def demo_preload_small_dataset():
    """Demo: Preload a small dataset."""
    print("\n" + "="*60)
    print("DEMO 2: Preload Small Dataset (PneumoniaMNIST)")
    print("="*60)
    
    server = VisionRAGMCPServer(
        collection_name="preload_demo",
        persist_directory="./chroma_db_preload_demo",
        image_store_dir="./image_store_preload_demo",
    )
    
    print("\nPreloading PneumoniaMNIST (train split, max 100 images)...")
    
    result = await server.preload_dataset(
        dataset_name="PneumoniaMNIST",
        split="train",
        max_images=100,  # Limit to 100 images for quick demo
    )
    
    if not result:
        print("❌ Error: Server preload returned no data")
        return

    if result.get("success"):
        print(f"\n✅ Success!")
        print(f"   Dataset: {result['dataset_name']}")
        print(f"   Split: {result['split']}")
        print(f"   Images loaded: {result['images_loaded']}")
        print(f"   Total embeddings in store: {result['total_embeddings']}")
        print(f"   Message: {result['message']}")
    else:
        print(f"\n❌ Error: {result.get('error')}")


async def demo_preload_multiple_datasets():
    """Demo: Preload multiple datasets."""
    print("\n" + "="*60)
    print("DEMO 3: Preload Multiple Datasets")
    print("="*60)
    
    server = VisionRAGMCPServer(
        collection_name="preload_demo_multi",
        persist_directory="./chroma_db_preload_demo_multi",
        image_store_dir="./image_store_preload_demo_multi",
    )
    
    datasets_to_load = [
        ("BreastMNIST", "train", 50),
        ("BloodMNIST", "train", 50),
        ("TissueMNIST", "test", 30),
    ]
    
    total_loaded = 0
    
    for dataset_name, split, max_images in datasets_to_load:
        print(f"\n📦 Loading {dataset_name} ({split}, max {max_images} images)...")
        
        result = await server.preload_dataset(
            dataset_name=dataset_name,
            split=split,
            max_images=max_images,
        )
        
        if not result:
            print(f"   ❌ Error: Server returned no data for {dataset_name}")
            continue

        if result.get("success"):
            total_loaded += result['images_loaded']
            print(f"   ✅ Loaded {result['images_loaded']} images")
        else:
            print(f"   ❌ Error: {result.get('error')}")
    
    # Get final statistics
    stats = await server.get_statistics()
    if not stats:
        print("❌ Error: Unable to retrieve statistics")
        return
    
    print(f"\n" + "="*60)
    print(f"Final Statistics:")
    print(f"   Total images loaded: {total_loaded}")
    print(f"   Total embeddings in store: {stats['total_embeddings']}")
    print(f"   Image store directory: {stats['image_store_directory']}")
    print("="*60)


async def demo_preload_and_search():
    """Demo: Preload dataset and perform search."""
    print("\n" + "="*60)
    print("DEMO 4: Preload and Search")
    print("="*60)
    
    server = VisionRAGMCPServer(
        collection_name="preload_demo_search",
        persist_directory="./chroma_db_preload_demo_search",
        image_store_dir="./image_store_preload_demo_search",
    )
    
    # Preload dataset
    print("\n1️⃣  Preloading PathMNIST (train split, max 200 images)...")
    
    preload_result = await server.preload_dataset(
        dataset_name="PathMNIST",
        split="train",
        max_images=200,
    )
    
    if not preload_result or not preload_result.get("success"):
        print(f"❌ Preload failed: {preload_result.get('error')}")
        return
    
    print(f"   ✅ Loaded {preload_result['images_loaded']} images")
    
    # Search by label
    print("\n2️⃣  Searching for images with label 0...")
    
    search_result = await server.search_by_label(label=0, n_results=5)
    if not search_result:
        print("❌ Error: Search did not return a result")
        return
    
    print(f"\n   Found {search_result.get('count', 0)} images with label 0:")
    print(f"   Human-readable label: {search_result.get('human_readable_label', 'Unknown')}")
    ids = search_result.get('ids', []) or []
    print(f"   Image IDs: {ids[:3]}...")  # Show first 3
    
    # List labels
    print("\n3️⃣  Available labels in PathMNIST:")
    
    # PathMNIST has different labels than OrganSMNIST
    # We need to get labels from the loaded dataset metadata
    for idx, metadata in enumerate(search_result.get('metadatas', [])[:5]):
        print(f"   Image {idx}: label={metadata.get('label')}, dataset={metadata.get('dataset')}")


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("Vision RAG Dataset Preloading Demonstrations")
    print("="*60)
    
    # Demo 1: List datasets
    await demo_list_datasets()
    
    # Demo 2: Preload small dataset
    await demo_preload_small_dataset()
    
    # Demo 3: Preload multiple datasets
    await demo_preload_multiple_datasets()
    
    # Demo 4: Preload and search
    await demo_preload_and_search()
    
    print("\n" + "="*60)
    print("All demonstrations completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())