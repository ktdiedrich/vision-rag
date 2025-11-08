#!/usr/bin/env python3
"""
Verification script to test the vision-RAG project structure.

This script verifies that all modules can be imported and basic
functionality works without requiring network access.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import tempfile
import shutil


def verify_imports():
    """Verify all modules can be imported."""
    print("✓ Testing module imports...")
    try:
        from vision_rag import (
            download_medmnist,
            load_organmnist_data,
            CLIPImageEncoder,
            ChromaRAGStore,
            ImageSearcher,
        )
        print("  ✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def verify_rag_store():
    """Verify ChromaDB RAG store works."""
    print("\n✓ Testing ChromaDB RAG store...")
    try:
        from vision_rag.rag_store import ChromaRAGStore
        
        temp_dir = tempfile.mkdtemp()
        
        # Create store
        rag_store = ChromaRAGStore(
            collection_name="verification_test",
            persist_directory=temp_dir,
        )
        
        # Create dummy embeddings
        embeddings = np.random.randn(5, 512).astype(np.float32)
        metadatas = [{"index": i} for i in range(5)]
        
        # Add embeddings
        rag_store.add_embeddings(embeddings, metadatas=metadatas)
        
        # Verify count
        assert rag_store.count() == 5, "Count mismatch"
        
        # Test search
        query = embeddings[0]
        results = rag_store.search(query, n_results=3)
        
        # Verify results
        assert len(results["ids"]) == 3, "Search results count mismatch"
        assert results["ids"][0] == "img_0", "Closest match not found"
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print("  ✓ RAG store working correctly")
        return True
    except Exception as e:
        print(f"  ✗ RAG store test failed: {e}")
        return False


def verify_data_utilities():
    """Verify data utility functions."""
    print("\n✓ Testing data utilities...")
    try:
        from vision_rag.data_loader import get_image_from_array
        
        # Create test image
        image_array = np.random.randint(0, 255, size=(28, 28, 3), dtype=np.uint8)
        
        # Convert to PIL Image
        pil_image = get_image_from_array(image_array)
        
        # Verify
        assert isinstance(pil_image, Image.Image), "Not a PIL Image"
        assert pil_image.size == (28, 28), "Wrong size"
        
        print("  ✓ Data utilities working correctly")
        return True
    except Exception as e:
        print(f"  ✗ Data utilities test failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Vision-RAG Project Verification")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(verify_imports())
    results.append(verify_rag_store())
    results.append(verify_data_utilities())
    
    # Summary
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All verification tests passed!")
        print("=" * 60)
        return 0
    else:
        print("✗ Some verification tests failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
