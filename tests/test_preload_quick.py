#!/usr/bin/env python3
"""
Quick MCP preload test - simulates Claude Desktop calling preload_dataset.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vision_rag.mcp_server import VisionRAGMCPServer


async def test_preload_scenarios():
    """Test various preload scenarios."""
    
    print("=" * 70)
    print("🧪 MCP Server Preload Test Suite")
    print("=" * 70)
    
    server = VisionRAGMCPServer()
    
    # Get initial stats
    stats = await server.get_statistics()
    print(f"\n📊 Initial state: {stats['total_embeddings']} embeddings\n")
    
    # Test 1: Small single-label dataset
    print("Test 1: Small single-label dataset (PneumoniaMNIST, 5 images)")
    print("-" * 70)
    result1 = await server.preload_dataset(
        dataset_name="PneumoniaMNIST",
        split="train",
        max_images=5,
        size=28
    )
    if result1.get("success"):
        print(f"✅ PASS - Loaded {result1['images_loaded']} images")
    else:
        print(f"❌ FAIL - {result1.get('error')}")
    
    # Test 2: Multi-label dataset
    print("\nTest 2: Multi-label dataset (ChestMNIST, 5 images)")
    print("-" * 70)
    result2 = await server.preload_dataset(
        dataset_name="ChestMNIST",
        split="train",
        max_images=5,
        size=28
    )
    if result2.get("success"):
        print(f"✅ PASS - Loaded {result2['images_loaded']} images")
    else:
        print(f"❌ FAIL - {result2.get('error')}")
    
    # Test 3: Different split
    print("\nTest 3: Test split (OrganSMNIST, 3 images)")
    print("-" * 70)
    result3 = await server.preload_dataset(
        dataset_name="OrganSMNIST",
        split="test",
        max_images=3,
        size=28
    )
    if result3.get("success"):
        print(f"✅ PASS - Loaded {result3['images_loaded']} images")
    else:
        print(f"❌ FAIL - {result3.get('error')}")
    
    # Test 4: List available datasets
    print("\nTest 4: List available datasets")
    print("-" * 70)
    datasets = await server.list_available_datasets()
    print(f"✅ PASS - Found {datasets['count']} datasets")
    print(f"   Examples: {list(datasets['datasets'].keys())[:3]}")
    
    # Test 5: Search by label
    print("\nTest 5: Search by label")
    print("-" * 70)
    search_result = await server.search_by_label(label=0, n_results=3)
    print(f"✅ PASS - Found {search_result['count']} images with label {search_result['label']}")
    
    # Final stats
    final_stats = await server.get_statistics()
    print("\n" + "=" * 70)
    print("📊 Final Statistics")
    print("=" * 70)
    print(f"Total embeddings: {final_stats['total_embeddings']}")
    print(f"Total images: {final_stats['total_images']}")
    print(f"ChromaDB: {final_stats['persist_directory']}")
    print(f"Image store: {final_stats['image_store_directory']}")
    
    # Summary
    tests_passed = sum([
        result1.get("success", False),
        result2.get("success", False),
        result3.get("success", False),
        datasets.get("count", 0) > 0,
        search_result.get("count", 0) > 0,
    ])
    
    print("\n" + "=" * 70)
    print(f"✅ Tests passed: {tests_passed}/5")
    print("=" * 70)
    
    if tests_passed == 5:
        print("\n🎉 All tests passed! MCP server is working correctly.")
        print("\nReady for Claude Desktop integration!")
        return True
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_preload_scenarios())
    sys.exit(0 if success else 1)
