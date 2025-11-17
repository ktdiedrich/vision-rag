#!/usr/bin/env python3
"""
Simple demo showing how to use the RAG visualization functionality.

This demo demonstrates how to:
1. Save sample input images going into the RAG store
2. Save sample search query images 
3. Save search results with retrieved images
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision_rag import (
    load_medmnist_data,
    get_image_from_array,
    get_human_readable_label,
    CLIPImageEncoder,
    ChromaRAGStore,
    ImageSearcher,
    RAGVisualizer,
    ImageFileStore,
    CLIP_MODEL_NAME
)


def main():
    """Run a simple visualization demo."""
    
    print("📊 Simple RAG Visualization Demo")
    print("=" * 40)
    
    # Initialize components
    visualizer = RAGVisualizer(output_dir="./output/simple_visualizations")
    print(f"Visualizations will be saved to: {visualizer.output_dir}")
    
    # Load small subset of data for quick demo
    print("\n📥 Loading OrganSMNIST data...")
    train_images, train_labels = load_medmnist_data(dataset_name="OrganSMNIST", split="train")
    test_images, test_labels = load_medmnist_data(dataset_name="OrganSMNIST", split="test")
    
    # Use small subsets for demonstration
    train_subset = [(train_images[i], int(train_labels[i])) for i in range(50)]
    test_subset = [(test_images[i], int(test_labels[i])) for i in range(5)]
    
    # 1. VISUALIZE INPUT IMAGES GOING INTO RAG STORE
    print("\n🖼️  Step 1: Visualizing input images for RAG store...")
    input_images = [get_image_from_array(img) for img, _ in train_subset[:20]]
    input_labels = [label for _, label in train_subset[:20]]
    
    input_path = visualizer.save_input_images_grid(
        images=input_images,
        labels=input_labels,
        filename="input_to_rag_store.png",
        title="Sample Images Added to RAG Store"
    )
    print(f"✅ Saved: {input_path}")
    
    # 2. SET UP RAG SYSTEM
    print("\n🧠 Step 2: Setting up RAG system...")
    encoder = CLIPImageEncoder()
    rag_store = ChromaRAGStore(collection_name="simple_demo", persist_directory="./chroma_db_simple")
    rag_store.clear()
    
    # Initialize image store
    image_store = ImageFileStore(storage_dir="./image_store_simple")
    image_store.clear()
    
    # Encode and add training images
    train_pil_images = [get_image_from_array(img) for img, _ in train_subset]
    train_embeddings = encoder.encode_images(train_pil_images)
    
    # Save images to disk and create metadata with paths
    train_metadata = []
    for i, (image, (_, label)) in enumerate(zip(train_pil_images, train_subset)):
        image_path = image_store.save_image(image, prefix="train")
        train_metadata.append({"label": label, "image_path": image_path})
    
    rag_store.add_embeddings(train_embeddings, metadatas=train_metadata)
    
    searcher = ImageSearcher(encoder=encoder, rag_store=rag_store)
    print(f"Added {len(train_subset)} images to RAG store")
    print(f"Saved {image_store.count()} images to disk")
    
    # 3. VISUALIZE EMBEDDING SPACE WITH T-SNE
    print("\n🌌 Step 3: Visualizing embedding space with t-SNE...")
    embedding_labels = [label for _, label in train_subset]
    embedding_viz_path = visualizer.save_embedding_space_visualization(
        embeddings=train_embeddings,
        labels=embedding_labels,
        method='tsne',
        filename="embedding_space_tsne.png"
    )
    print(f"✅ Saved: {embedding_viz_path}")
    
    # 4. VISUALIZE SEARCH INPUT IMAGES
    print("\n🔍 Step 4: Visualizing search query images...")
    query_images = [get_image_from_array(img) for img, _ in test_subset]
    query_labels = [label for _, label in test_subset]
    
    search_input_path = visualizer.save_search_input_images(
        images=query_images,
        labels=query_labels,
        filename="search_query_images.png",
        title="Search Query Images"
    )
    print(f"✅ Saved: {search_input_path}")
    
    # 5. PERFORM SEARCHES AND VISUALIZE RESULTS
    print("\n🎯 Step 5: Performing searches and visualizing results...")

    for i, (query_img, query_label) in enumerate(zip(query_images, query_labels)):
        query_name = get_human_readable_label(query_label)
        print(f"\n  Search {i+1}: Query with label {query_label} ({query_name})")
        
        # Perform search
        results = searcher.search(query_img, n_results=5)
        
        # Get retrieved images
        retrieved_images = []
        for result_id in results['ids']:
            # Parse ID to get training subset index
            subset_idx = int(result_id.split('_')[1])
            original_img = get_image_from_array(train_subset[subset_idx][0])
            retrieved_images.append(original_img)
        
        # Save search results visualization
        results_path = visualizer.save_search_results(
            query_image=query_img,
            query_label=query_label,
            retrieved_images=retrieved_images,
            retrieved_metadata=results['metadatas'],
            distances=results['distances'],
            filename=f"search_results_{i+1}.png",
            title=f"Search Results {i+1}: {query_name} ({query_label})"
        )
        print(f"  ✅ Saved: {results_path}")
        
        # Print summary
        retrieved_labels = [meta['label'] for meta in results['metadatas']]
        retrieved_names = [get_human_readable_label(label) for label in retrieved_labels]
        distances = [f"{d:.3f}" for d in results['distances']]
        print(f"  Retrieved: {[f'{name} ({label})' for name, label in zip(retrieved_names, retrieved_labels)]}")
        print(f"  Distances: {distances}")
    
    print(f"\n🎉 Demo complete!")
    print(f"📁 All visualizations saved to: {visualizer.output_dir.absolute()}")
    print("\nGenerated files:")
    for viz_file in sorted(visualizer.output_dir.glob("*.png")):
        print(f"  - {viz_file.name}")


if __name__ == "__main__":
    main()
