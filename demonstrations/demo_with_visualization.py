#!/usr/bin/env python3
"""
Demonstration script for vision RAG system with visualization.

This script demonstrates the full pipeline:
1. Loading and visualizing input data
2. Adding images to RAG store with visualization
3. Performing searches with input/output visualization
4. Creating various analysis visualizations
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from vision_rag import (
    load_medmnist_data,
    get_image_from_array,
    get_human_readable_label,
    build_encoder,
    ChromaRAGStore,
    ImageSearcher,
    RAGVisualizer,
    ImageFileStore,
)
from vision_rag.config import ENCODER_TYPE, DINO_MODEL_NAME, NEAREST_NEIGHBORS, MEDMNIST_DATASET, LARGE_SUBSET


def main():
    """Run the vision RAG demonstration with visualizations."""
    
    print("🔬 Vision RAG System Demonstration with Visualizations")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = RAGVisualizer(output_dir="./output/visualizations")
    print(f"📊 Visualizations will be saved to: {visualizer.output_dir}")
    
    # Step 1: Load data
    print(f"\n📥 Loading {MEDMNIST_DATASET} data...")
    train_images, train_labels = load_medmnist_data(dataset_name=MEDMNIST_DATASET, split="train")
    test_images, test_labels = load_medmnist_data(dataset_name=MEDMNIST_DATASET, split="test")
    
    print(f"   Training set: {len(train_images)} images")
    print(f"   Test set: {len(test_images)} images")
    
    # Step 2: Visualize sample input images that will go into RAG store
    print("\n🖼️  Visualizing sample input images for RAG store...")
    sample_indices = np.random.choice(len(train_images), size=20, replace=False)
    sample_train_images = [train_images[i] for i in sample_indices]
    sample_train_labels = [int(train_labels[i]) for i in sample_indices]
    
    input_viz_path = visualizer.save_input_images_grid(
        images=sample_train_images,
        labels=sample_train_labels,
        filename="01_input_images_to_rag.png",
        title="Sample Training Images Added to RAG Store"
    )
    print(f"   ✅ Saved input visualization: {input_viz_path}")
    
    # Step 3: Visualize label distribution
    print("\n📊 Visualizing label distribution...")
    label_dist_path = visualizer.save_label_distribution(
        labels=[int(label) for label in train_labels],
        filename="02_label_distribution.png"
    )
    print(f"   ✅ Saved label distribution: {label_dist_path}")
    
    # Step 4: Initialize encoder and encode subset of training images
    print("\n🧠 Initializing encoder via build_encoder() using configured ENCODER_TYPE...")
    encoder = build_encoder(encoder_type="dino")  # uses ENCODER_TYPE from config; set VISION_RAG_ENCODER='dino' to use DINO
    if ENCODER_TYPE and ENCODER_TYPE.lower().startswith("dino"):
        print(f"   Using DINO model: {DINO_MODEL_NAME}")
    print(f"   Embedding dimension: {encoder.embedding_dimension}")
    
    # Use a smaller subset for demonstration to speed up processing
    subset_indices = np.random.choice(len(train_images), size=LARGE_SUBSET, replace=False)
    subset_images = [get_image_from_array(train_images[i]) for i in subset_indices]
    subset_labels = [int(train_labels[i]) for i in subset_indices]
    
    print(f"\n🔢 Encoding {len(subset_images)} training images...")
    train_embeddings = encoder.encode_images(subset_images)
    print(f"   Embeddings shape: {train_embeddings.shape}")
    
    # Step 5: Create and populate RAG store
    print("\n🗃️  Creating RAG store...")
    rag_store = ChromaRAGStore(
        collection_name="mnist_demo",
        persist_directory="./chroma_db_demo",
    )
    
    # Clear any existing data
    rag_store.clear()
    
    # Initialize image store
    image_store = ImageFileStore(storage_dir="./image_store_demo")
    image_store.clear()
    
    # Save images to disk and add embeddings with metadata including paths
    print(f"\n💾 Saving images to disk...")
    metadatas = []
    for i, (image, label) in enumerate(zip(subset_images, subset_labels)):
        image_path = image_store.save_image(image, prefix="train")
        metadatas.append({"label": label, "image_path": image_path})
    
    rag_store.add_embeddings(train_embeddings, metadatas=metadatas)
    print(f"   Added {rag_store.count()} embeddings to RAG store")
    print(f"   Saved {image_store.count()} images to disk")
    
    # Step 6: Visualize embedding space
    print("\n🌌 Visualizing embedding space...")
    embedding_viz_path = visualizer.save_embedding_space_visualization(
        embeddings=train_embeddings,
        labels=subset_labels,
        method='tsne',
        filename="03_embedding_space_tsne.png",
        model_name=getattr(encoder, "model_name", None),
    )
    print(f"   ✅ Saved embedding space visualization: {embedding_viz_path}")
    
    # Step 7: Perform searches and visualize results
    print("\n🔍 Performing searches and visualizing results...")
    searcher = ImageSearcher(encoder=encoder, rag_store=rag_store)
    
    # Select some test images for search queries
    query_indices = np.random.choice(len(test_images), size=5, replace=False)
    query_images = [get_image_from_array(test_images[i]) for i in query_indices]
    query_labels = [int(test_labels[i]) for i in query_indices]
    
    # Save search input images
    search_input_path = visualizer.save_search_input_images(
        images=query_images,
        labels=query_labels,
        filename="04_search_input_images.png"
    )
    print(f"   ✅ Saved search input visualization: {search_input_path}")
    
    # Perform searches, classify queries, and save results
    classification_results = []
    for i, (query_img, query_label) in enumerate(zip(query_images, query_labels)):
        readable_query_label = get_human_readable_label(query_label, dataset_name=MEDMNIST_DATASET)
        print(f"\n   🔍 Search {i+1}: Query image with {readable_query_label}")
        
        results = searcher.search(query_img, n_results=NEAREST_NEIGHBORS)
        
        # Get retrieved images for visualization
        retrieved_images = []
        for result_id in results['ids']:  # results are already flattened by RAG store
            # Parse the ID to get the index (format is 'img_N')
            id_index = int(result_id.split('_')[1])
            original_index = subset_indices[id_index]
            retrieved_img = get_image_from_array(train_images[original_index])
            retrieved_images.append(retrieved_img)
        
        # Save search results
        search_results_path = visualizer.save_search_results(
            query_image=query_img,
            query_label=query_label,
            retrieved_images=retrieved_images,
            retrieved_metadata=results['metadatas'],
            distances=results['distances'],
            filename=f"05_search_results_{i+1}.png",
            title=f"Search Results {i+1}: Query Label {query_label}"
        )
        print(f"      ✅ Saved search results: {search_results_path}")
        
        # Print summary with human readable labels
        retrieved_labels = [meta.get('label', meta.get('index', 'unknown')) for meta in results['metadatas']]
        readable_retrieved_labels = [get_human_readable_label(label, dataset_name=MEDMNIST_DATASET) if isinstance(label, int) else str(label) for label in retrieved_labels]
        print(f"      Retrieved labels: {readable_retrieved_labels}")
        print(f"      Distances: {[f'{d:.3f}' for d in results['distances']]}")

        # Classify the query image using k-NN majority vote and compare to ground-truth
        classification = searcher.classify(query_img, n_results=NEAREST_NEIGHBORS)
        predicted_label = classification.get("label")
        confidence = classification.get("confidence")
        predicted_human = get_human_readable_label(predicted_label, dataset_name=MEDMNIST_DATASET) if isinstance(predicted_label, int) else str(predicted_label)
        truth_human = get_human_readable_label(query_label, dataset_name=MEDMNIST_DATASET)
        is_correct = predicted_label == query_label
        print(f"      Classification: Predicted {predicted_human} ({predicted_label}), Confidence: {confidence:.2f}; Truth: {truth_human} ({query_label}) -> {'CORRECT' if is_correct else 'WRONG'}")
        classification_results.append((query_label, predicted_label, confidence))
    
    # Compute classification accuracy
    correct = sum(1 for truth, pred, conf in classification_results if truth == pred)
    total = len(classification_results)
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n🎉 Demonstration complete!")
    print(f"\n🧮 Classification summary: {correct}/{total} correct ({accuracy:.2%} accuracy)")
    print(f"📁 All visualizations saved to: {visualizer.output_dir.absolute()}")
    print("\nGenerated files:")
    for viz_file in sorted(visualizer.output_dir.glob("*.png")):
        print(f"   - {viz_file.name}")
    

if __name__ == "__main__":
    main()