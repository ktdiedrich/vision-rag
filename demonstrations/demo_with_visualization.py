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
    DICOMClassifier,
    get_dataset_config,
)
from vision_rag.config import ENCODER_TYPE, DINO_MODEL_NAME, NEAREST_NEIGHBORS, MEDMNIST_DATASET, LARGE_SUBSET
import os
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
    val_images, val_labels = load_medmnist_data(dataset_name=MEDMNIST_DATASET, split="val")
    test_images, test_labels = load_medmnist_data(dataset_name=MEDMNIST_DATASET, split="test")
    
    print(f"   Training set: {len(train_images)} images")
    print(f"   Validation set: {len(val_images)} images")
    print(f"   Test set: {len(test_images)} images")

    def verify_checkpoint_dir(ckpt_path: str, strict: bool = False) -> bool:
        """Verify the HF checkpoint directory contains common files and raise if missing.

        This helper checks locally-saved checkpoints only. If `ckpt_path` is a
        remote HF repo id (not a local dir) the function prints a message and
        returns without asserting.
        """
        ckpt_p = Path(ckpt_path)
        if not ckpt_p.exists():
            print(f"   ⚠️ Checkpoint path {ckpt_path} does not exist locally; cannot verify files.")
            return False
        # Basic expected files from transformers' Trainer.save_model and image processors
        expected_any = ["pytorch_model.bin", "pytorch_model.safetensors"]
        expected_all = ["config.json", "training_args.bin", "preprocessor_config.json"]
        found_any = any((ckpt_p / fname).exists() for fname in expected_any)
        missing = [f for f in expected_all if not (ckpt_p / f).exists()]
        if not found_any:
            msg = f"Expected one of {expected_any} in checkpoint directory {ckpt_path} but none were found."
            if strict:
                raise AssertionError(msg)
            else:
                print(f"   ⚠️ {msg}")
                return False
        if missing:
            msg = f"Missing expected checkpoint files in {ckpt_path}: {missing}"
            if strict:
                raise AssertionError(msg)
            else:
                print(f"   ⚠️ {msg}")
                return False
        # Print the files we found
        print(f"   ✅ Verified checkpoint at {ckpt_path} contains expected files:")
        for fname in expected_any + expected_all:
            fpath = ckpt_p / fname
            if fpath.exists():
                print(f"     - {fpath}")
        return True
    
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
    
    # Optionally train a DINO classifier before using the model as an encoder.
    # To enable, set environment variable VISION_RAG_TRAIN_DINO=true
    # Allow either using an existing checkpoint path or training a new checkpoint
    train_dino_classifier = os.getenv("VISION_RAG_TRAIN_DINO", "false").lower() in ("1", "true", "yes")
    default_ckpt_dir = "./dino_finetuned"
    ckpt_env = os.getenv("VISION_RAG_DINO_CHECKPOINT")
    ckpt_dir = ckpt_env if ckpt_env else default_ckpt_dir
    # Ensure encoder is always defined in this local scope for fallback checks
    encoder = None
    # If a checkpoint directory is present and the user didn't request training,
    # load the fine-tuned checkpoint directly for embeddings.
    # If training is not requested and either a local default checkpoint exists
    # or the user provided a checkpoint/model name via environment, attempt
    # to load it for embeddings.
    if not train_dino_classifier and (ckpt_env is not None or os.path.isdir(ckpt_dir)):
        print(f"\n⚡ Found existing fine-tuned checkpoint or model '{ckpt_dir}', attempting to load it for embeddings...")
        try:
            # If checkpoint is a local path, verify expected files exist
            if os.path.isdir(ckpt_dir):
                verify_checkpoint_dir(ckpt_dir, strict=False)
            encoder = build_encoder(encoder_type="dino", model_name=ckpt_dir, ignore_mismatched_sizes=True)
            if ENCODER_TYPE and ENCODER_TYPE.lower().startswith("dino"):
                print(f"   Using fine-tuned DINO model from: {ckpt_dir}")
            print(f"   Embedding dimension: {encoder.embedding_dimension}")
        except Exception as e:
            print(f"   ⚠️ Failed to load checkpoint {ckpt_dir}: {e}. Falling back to base pre-trained DINO model.")
            train_dino_classifier = False
            encoder = None

    if train_dino_classifier:
        print("\n⚡ Training a DINO classifier on a small subset before using the model for embeddings...")
        num_classes = get_dataset_config(MEDMNIST_DATASET)["n_classes"]
        dicom_classifier = DICOMClassifier(model_name=DINO_MODEL_NAME, num_labels=num_classes)
        # Use a small portion for fast demo training
        train_n = min(200, len(train_images))
        small_train_imgs = [get_image_from_array(train_images[i]) for i in range(train_n)]
        small_train_lbls = [int(train_labels[i]) for i in range(train_n)]
        val_n = min(50, len(train_images) - train_n)
        val_imgs, val_lbls = None, None
        if val_n > 0:
            val_imgs = [get_image_from_array(train_images[i + train_n]) for i in range(val_n)]
            val_lbls = [int(train_labels[i + train_n]) for i in range(val_n)]
        _, ckpt_dir = dicom_classifier.fit(
            small_train_imgs,
            small_train_lbls,
            val_images=val_imgs,
            val_labels=val_lbls,
            output_dir="./dino_finetuned",
            num_train_epochs=1,
            per_device_train_batch_size=8,
            logging_steps=10,
        )
        print(f"   ✅ DINO classifier trained and saved to: {ckpt_dir}")
        # Verify the saved checkpoint contains expected artifacts (strict)
        verify_checkpoint_dir(ckpt_dir, strict=True)
        encoder = build_encoder(encoder_type="dino", model_name=ckpt_dir, ignore_mismatched_sizes=True)
        if ENCODER_TYPE and ENCODER_TYPE.lower().startswith("dino"):
            print(f"   Using fine-tuned DINO model: {ckpt_dir}")
        print(f"   Embedding dimension: {encoder.embedding_dimension}")
    else:
        # Step 4: Initialize encoder and encode subset of training images
        print("\n🧠 Initializing encoder via build_encoder() using configured ENCODER_TYPE...")
        if encoder is None:
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

    # Compute continuous scores for ROC/PR (binary case: treat label=1 as positive)
    # For k-NN we estimate the positive-class score as fraction of neighbors that are positive.
    # For predictions with predicted_label == 1, use confidence; otherwise use 1 - confidence.
    y_true = eval_true_labels_list
    y_scores = []
    for r in eval_results:
        pred = r.get("predicted_label")
        conf = r.get("confidence", 0.0)
        if pred == 1:
            score = conf
        else:
            score = 1.0 - conf
        y_scores.append(score)

    # Plot ROC and PR curves for the LARGE_SUBSET evaluation
    roc_png = visualizer.plot_roc_curve(y_true, y_scores, pos_label=1, filename="08_roc_curve_large.png")
    pr_png = visualizer.plot_precision_recall_curve(y_true, y_scores, pos_label=1, filename="09_pr_curve_large.png")
    print(f"   ✅ Saved large ROC curve plot: {roc_png}")
    print(f"   ✅ Saved large Precision-Recall plot: {pr_png}")
    

if __name__ == "__main__":
    main()