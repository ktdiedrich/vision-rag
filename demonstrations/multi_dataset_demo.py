"""
Demonstration of using multiple MedMNIST datasets with Vision RAG.

This example shows how to:
1. List available datasets
2. Use different datasets (PathMNIST and ChestMNIST)
3. Compare results across datasets
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision_rag import (
    list_available_datasets,
    get_dataset_config,
    download_medmnist,
    load_medmnist_data,
    get_medmnist_label_names,
    build_encoder,
    ChromaRAGStore,
    ImageSearcher,
    ImageFileStore,
)


def main():
    print("=" * 80)
    print("Vision RAG - Multi-Dataset Demonstration")
    print("=" * 80)
    
    # 1. List all available datasets
    print("\n1. Available MedMNIST Datasets:")
    print("-" * 40)
    datasets = list_available_datasets()
    for dataset in datasets:
        config = get_dataset_config(dataset)
        print(f"  • {dataset:15s} - {config['description']}")
        print(f"    {config['n_classes']} classes, {config['channels']} channel(s)")
    
    # 2. Work with PathMNIST dataset
    print("\n2. Working with PathMNIST (Colon Pathology)")
    print("-" * 40)
    
    # Download and load PathMNIST
    print("Downloading PathMNIST dataset...")
    download_medmnist(dataset_name="PathMNIST", root="./data")
    
    print("Loading training data...")
    train_images, train_labels = load_medmnist_data(
        dataset_name="PathMNIST",
        split="train",
        root="./data"
    )
    
    # Get label information
    label_names = get_medmnist_label_names(dataset_name="PathMNIST", root="./data")
    print(f"PathMNIST has {len(label_names)} classes:")
    for label_id, label_name in label_names.items():
        print(f"  {label_id}: {label_name}")
    
    print(f"\nLoaded {len(train_images)} training images")
    print(f"Image shape: {train_images[0].shape} (RGB)")
    
    # 3. Build RAG store for PathMNIST
    print("\n3. Building RAG Store for PathMNIST")
    print("-" * 40)
    
    # Initialize encoder
    print("Initializing CLIP encoder...")
    encoder = build_encoder(encoder_type="clip")
    
    # Take a subset for demonstration (first 100 images)
    print("Encoding first 100 images...")
    subset_images = train_images[:100]
    subset_labels = train_labels[:100]
    embeddings = encoder.encode_images([img for img in subset_images])
    
    # Create RAG store and image store
    print("Creating RAG store and image store...")
    rag_store = ChromaRAGStore(
        collection_name="pathmnist_demo",
        persist_directory="./chroma_db_multi_demo",
    )
    image_store = ImageFileStore(storage_dir="./image_store_multi_demo")
    
    # Save images and add embeddings
    metadatas = []
    for i, (img, label) in enumerate(zip(subset_images, subset_labels)):
        image_path = image_store.save_image(img, prefix="path")
        metadatas.append({"label": int(label), "dataset": "PathMNIST", "image_path": image_path})
    
    rag_store.add_embeddings(embeddings, metadatas=metadatas)
    print(f"Added {rag_store.count()} embeddings to RAG store")
    print(f"Saved {image_store.count()} images to disk")
    
    # 4. Search for similar images
    print("\n4. Searching for Similar Images")
    print("-" * 40)
    
    # Load test data
    test_images, test_labels = load_medmnist_data(
        dataset_name="PathMNIST",
        split="test",
        root="./data"
    )
    
    # Search with first test image
    searcher = ImageSearcher(encoder=encoder, rag_store=rag_store)
    query_image = test_images[0]
    query_label = test_labels[0]
    
    print(f"Query image label: {label_names.get(int(query_label), 'Unknown')}")
    
    results = searcher.search(query_image, n_results=5)
    print(f"\nFound {len(results['ids'])} similar images:")
    for idx, (result_id, distance, metadata) in enumerate(
        zip(results["ids"], results["distances"], results["metadatas"])
    ):
        result_label = metadata.get("label", "Unknown")
        result_name = label_names.get(int(result_label), "Unknown")
        print(f"  {idx + 1}. ID: {result_id}, Distance: {distance:.4f}, Label: {result_name}")
    
    # 5. Demonstrate with ChestMNIST
    print("\n5. Quick Demo with ChestMNIST (Chest X-rays)")
    print("-" * 40)
    
    # Get ChestMNIST configuration
    chest_config = get_dataset_config("ChestMNIST")
    print(f"ChestMNIST has {chest_config['n_classes']} classes")
    print(f"Description: {chest_config['description']}")
    print(f"Image channels: {chest_config['channels']} (grayscale)")
    
    print("\n" + "=" * 80)
    print("Demonstration Complete!")
    print("=" * 80)
    print("\nTo use a different dataset, simply change the dataset_name parameter:")
    print("  download_medmnist(dataset_name='ChestMNIST')")
    print("  load_medmnist_data(dataset_name='ChestMNIST', split='train')")
    print("\nOr set the environment variable:")
    print("  export VISION_RAG_DATASET='ChestMNIST'")
    print("  python your_script.py")


if __name__ == "__main__":
    main()
