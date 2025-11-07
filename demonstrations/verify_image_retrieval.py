#!/usr/bin/env python3
"""
Verify that images can be retrieved from the image store using metadata.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision_rag import ChromaRAGStore, ImageFileStore


def main():
    print("🔍 Verifying Image Retrieval from Store")
    print("=" * 50)
    
    # Connect to the RAG store
    rag_store = ChromaRAGStore(
        collection_name="simple_example",
        persist_directory="./chroma_db_simple"
    )
    
    # Connect to the image store
    image_store = ImageFileStore(storage_dir="./image_store_simple")
    
    print(f"\n📊 RAG Store Statistics:")
    print(f"   Total embeddings: {rag_store.count()}")
    print(f"   Total images: {image_store.count()}")
    
    # Get first few items
    results = rag_store.collection.get(limit=5)
    
    print(f"\n🖼️  Sample Metadata (first 5 entries):")
    for i, (doc_id, metadata) in enumerate(zip(results['ids'], results['metadatas'])):
        print(f"\n   [{i+1}] ID: {doc_id}")
        print(f"       Label: {metadata.get('label', 'N/A')}")
        print(f"       Image Path: {metadata.get('image_path', 'N/A')}")
        
        # Verify image exists
        if 'image_path' in metadata:
            image_path = metadata['image_path']
            exists = image_store.image_exists(image_path)
            print(f"       Image Exists: {'✅ Yes' if exists else '❌ No'}")
            
            if exists:
                # Try to load the image
                try:
                    image = image_store.load_image(image_path)
                    print(f"       Image Size: {image.size}, Mode: {image.mode}")
                except Exception as e:
                    print(f"       ❌ Error loading: {e}")
    
    print("\n✅ Verification complete!")


if __name__ == "__main__":
    main()
