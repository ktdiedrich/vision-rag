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
import csv
import json


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
    
    # Select some test images for search query visualizations (5 images only)
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
        classification_results.append({
            "query_index": int(i),
            "query_label": int(query_label) if isinstance(query_label, (int, np.integer)) else query_label,
            "query_label_name": truth_human,
            "predicted_label": int(predicted_label) if isinstance(predicted_label, (int, np.integer)) else predicted_label,
            "predicted_label_name": predicted_human,
            "confidence": float(confidence) if confidence is not None else 0.0,
            "correct": bool(is_correct),
        })
    
    # Compute classification accuracy
    correct = sum(1 for row in classification_results if row.get("query_label") == row.get("predicted_label"))
    total = len(classification_results)
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n🎉 Demonstration complete!")
    print(f"\n🧮 Classification summary: {correct}/{total} correct ({accuracy:.2%} accuracy)")

    # Save classification results to CSV and JSON
    csv_path = visualizer.output_dir / "classification_results.csv"
    json_path = visualizer.output_dir / "classification_results.json"

    # CSV header order
    csv_fields = [
        "query_index",
        "query_label",
        "query_label_name",
        "predicted_label",
        "predicted_label_name",
        "confidence",
        "correct",
    ]

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()
        for row in classification_results:
            writer.writerow({k: row.get(k) for k in csv_fields})

    # Write JSON
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(classification_results, jf, indent=2)

    print(f"\n💾 Saved classification results to: {csv_path} and {json_path}")
    print(f"📁 All visualizations saved to: {visualizer.output_dir.absolute()}")
    print("\nGenerated files:")
    for viz_file in sorted(visualizer.output_dir.glob("*.png")):
        print(f"   - {viz_file.name}")
    # Automatically evaluate and produce confusion matrix/metrics plots
    print("\n🔢 Running evaluation and saving confusion matrix/metrics plots...")
    true_labels_eval = [row["query_label"] for row in classification_results]
    predicted_labels_eval = [row["predicted_label"] for row in classification_results]
    metrics = searcher.evaluate_classification(true_labels=true_labels_eval, predicted_labels=predicted_labels_eval, save_results=True, output_dir=str(visualizer.output_dir), filename_prefix="06_demo_evaluation", to_csv=True, to_json=True, to_csv_matrix=True)
    print(f"   ✅ Saved evaluation artifacts to: {metrics.get('saved_paths')}")
    # Create PNG plots
    cm_png = visualizer.plot_confusion_matrix(metrics["confusion"], labels=metrics["labels"], filename="06_confusion_matrix.png")
    per_label_png = visualizer.plot_per_label_metrics(metrics["per_label"], labels_order=metrics["labels"], filename="06_per_label_metrics.png")
    print(f"   ✅ Saved confusion matrix plot: {cm_png}")
    print(f"   ✅ Saved per-label metrics plot: {per_label_png}")
    # ------------------------------------------------------------------
    # Now evaluate a larger subset of test images (LARGE_SUBSET) and save metrics
    # ------------------------------------------------------------------
    print(f"\n🔎 Running full evaluation on {LARGE_SUBSET} test images and saving metrics/plots...")
    eval_indices = np.random.choice(len(test_images), size=min(LARGE_SUBSET, len(test_images)), replace=False)
    eval_images = [get_image_from_array(test_images[i]) for i in eval_indices]
    eval_true_labels = [int(test_labels[i]) for i in eval_indices]

    # Classify each eval image using k-NN and collect results
    eval_results = []
    for i, (img, lbl) in enumerate(zip(eval_images, eval_true_labels)):
        classification = searcher.classify(img, n_results=NEAREST_NEIGHBORS)
        predicted_label = classification.get("label")
        confidence = classification.get("confidence")
        eval_results.append({
            "index": int(i),
            "true_label": int(lbl) if isinstance(lbl, (int, np.integer)) else lbl,
            "predicted_label": int(predicted_label) if isinstance(predicted_label, (int, np.integer)) else predicted_label,
            "confidence": float(confidence) if confidence is not None else 0.0,
            "correct": bool(predicted_label == lbl),
        })

    # Save eval results as CSV and JSON
    eval_csv_path = visualizer.output_dir / "evaluation_large_results.csv"
    eval_json_path = visualizer.output_dir / "evaluation_large_results.json"
    eval_csv_fields = ["index", "true_label", "predicted_label", "confidence", "correct"]
    with open(eval_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=eval_csv_fields)
        writer.writeheader()
        for row in eval_results:
            writer.writerow({k: row.get(k) for k in eval_csv_fields})
    with open(eval_json_path, "w", encoding="utf-8") as jf:
        json.dump(eval_results, jf, indent=2)

    print(f"   ✅ Saved eval classification results to: {eval_csv_path} and {eval_json_path}")

    # Compute evaluation metrics and save using evaluate_classification with autosave
    eval_true_labels_list = [r["true_label"] for r in eval_results]
    eval_predicted_labels_list = [r["predicted_label"] for r in eval_results]
    eval_metrics = searcher.evaluate_classification(
        true_labels=eval_true_labels_list,
        predicted_labels=eval_predicted_labels_list,
        save_results=True,
        output_dir=str(visualizer.output_dir),
        filename_prefix="07_eval_large",
        to_csv=True,
        to_json=True,
        to_csv_matrix=True,
    )

    # Create plots for the large evaluation
    cm_large_png = visualizer.plot_confusion_matrix(eval_metrics["confusion"], labels=eval_metrics["labels"], filename="07_confusion_matrix_large.png")
    per_label_large_png = visualizer.plot_per_label_metrics(eval_metrics["per_label"], labels_order=eval_metrics["labels"], filename="07_per_label_metrics_large.png")
    print(f"   ✅ Saved large confusion matrix plot: {cm_large_png}")
    print(f"   ✅ Saved large per-label metrics plot: {per_label_large_png}")
    

if __name__ == "__main__":
    main()