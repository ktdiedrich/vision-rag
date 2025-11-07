"""Vision-RAG demo application."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision_rag import (
    download_organmnist,
    load_organmnist_data,
    CLIPImageEncoder,
    ChromaRAGStore,
    ImageSearcher,
)


def main():
    """Run the vision-RAG demo."""
    print("=" * 60)
    print("Vision-RAG: Medical Image Retrieval Demo")
    print("=" * 60)
    
    # Step 1: Download dataset
    print("\n[1/6] Downloading OrganSMNIST dataset...")
    download_organmnist(root="./data")
    print("✓ Dataset downloaded")
    
    # Step 2: Load training data (use subset for demo)
    print("\n[2/6] Loading training data...")
    train_images, train_labels = load_organmnist_data(split="train", root="./data")
    print(f"✓ Loaded {len(train_images)} training images")
    
    # Use a subset for faster demo (100 images)
    n_train = min(100, len(train_images))
    train_images_subset = train_images[:n_train]
    train_labels_subset = train_labels[:n_train]
    print(f"  Using {n_train} images for demo")
    
    # Step 3: Initialize encoder
    print("\n[3/6] Initializing CLIP encoder (clip-ViT-B-32)...")
    encoder = CLIPImageEncoder(model_name="clip-ViT-B-32")
    print(f"✓ Encoder initialized (embedding dim: {encoder.embedding_dimension})")
    
    # Step 4: Encode training images
    print("\n[4/6] Encoding training images...")
    train_embeddings = encoder.encode_images([img for img in train_images_subset])
    print(f"✓ Encoded {len(train_embeddings)} images")
    print(f"  Embedding shape: {train_embeddings.shape}")
    
    # Step 5: Create RAG store and add embeddings
    print("\n[5/6] Creating ChromaDB RAG store...")
    rag_store = ChromaRAGStore(
        collection_name="organmnist_demo",
        persist_directory="./chroma_db_main",
    )
    
    # Clear any existing data
    if rag_store.count() > 0:
        print("  Clearing existing data...")
        rag_store.clear()
    
    # Add embeddings with metadata
    metadatas = [{"label": int(label), "split": "train"} for label in train_labels_subset]
    rag_store.add_embeddings(train_embeddings, metadatas=metadatas)
    print(f"✓ Added {rag_store.count()} embeddings to RAG store")
    
    # Step 6: Search with test images
    print("\n[6/6] Searching with test images...")
    test_images, test_labels = load_organmnist_data(split="test", root="./data")
    print(f"✓ Loaded {len(test_images)} test images")
    
    # Create searcher
    searcher = ImageSearcher(encoder=encoder, rag_store=rag_store)
    
    # Search with first 3 test images
    n_test = min(3, len(test_images))
    print(f"\n  Searching for {n_test} test images...")
    
    for i in range(n_test):
        test_image = test_images[i]
        test_label = test_labels[i]
        
        results = searcher.search(test_image, n_results=5)
        
        print(f"\n  Test image {i} (label: {test_label}):")
        print(f"    Found {len(results['ids'])} similar images")
        print(f"    Top 3 matches:")
        for j in range(min(3, len(results['ids']))):
            img_id = results['ids'][j]
            distance = results['distances'][j]
            metadata = results['metadatas'][j]
            print(f"      {j+1}. ID: {img_id}, Distance: {distance:.4f}, Label: {metadata.get('label', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

