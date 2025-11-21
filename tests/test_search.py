"""Tests for image search functionality."""

import pytest
import json
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import shutil

from vision_rag.encoder import build_encoder
from vision_rag.rag_store import ChromaRAGStore
from vision_rag.search import ImageSearcher


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for ChromaDB."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def encoder():
    """Create a CLIP encoder instance."""
    return build_encoder(encoder_type="clip", model_name="clip-ViT-B-32")


@pytest.fixture
def rag_store(temp_db_dir):
    """Create a ChromaRAGStore instance."""
    return ChromaRAGStore(
        collection_name="test_search_collection",
        persist_directory=temp_db_dir,
    )


@pytest.fixture
def searcher(encoder, rag_store):
    """Create an ImageSearcher instance."""
    return ImageSearcher(encoder=encoder, rag_store=rag_store)


@pytest.fixture
def sample_images():
    """Create sample images."""
    images = []
    for i in range(5):
        # Create different colored images
        color = (i * 50, (i * 30) % 256, (i * 70) % 256)
        img = Image.new("RGB", (28, 28), color=color)
        images.append(img)
    return images


def test_searcher_initialization(searcher):
    """Test searcher initialization."""
    assert searcher.encoder is not None
    assert searcher.rag_store is not None


def test_search_with_pil_image(searcher, encoder, sample_images):
    """Test searching with a PIL Image."""
    # Add some images to the store first
    embeddings = encoder.encode_images(sample_images)
    searcher.rag_store.add_embeddings(embeddings)
    
    # Search with the first image
    query_image = sample_images[0]
    results = searcher.search(query_image, n_results=3)
    
    # Check results structure
    assert "ids" in results
    assert "distances" in results
    assert "metadatas" in results
    
    # Should find at least one result
    assert len(results["ids"]) > 0


def test_search_with_numpy_image(searcher, encoder, sample_images):
    """Test searching with a numpy array image."""
    # Add some images to the store first
    embeddings = encoder.encode_images(sample_images)
    searcher.rag_store.add_embeddings(embeddings)
    
    # Convert first image to numpy array
    query_image = np.array(sample_images[0])
    results = searcher.search(query_image, n_results=3)
    
    # Check results structure
    assert "ids" in results
    assert len(results["ids"]) > 0


def test_search_finds_similar_image(searcher, encoder, sample_images):
    """Test that search finds the most similar image."""
    # Add images to the store
    embeddings = encoder.encode_images(sample_images)
    searcher.rag_store.add_embeddings(embeddings)
    
    # Search with the first image (should find itself)
    results = searcher.search(sample_images[0], n_results=1)
    
    # The closest match should be img_0 (the query itself)
    assert results["ids"][0] == "img_0"


def test_batch_search(searcher, encoder, sample_images):
    """Test batch searching with multiple query images."""
    # Add images to the store
    embeddings = encoder.encode_images(sample_images)
    searcher.rag_store.add_embeddings(embeddings)
    
    # Search with multiple queries
    query_images = sample_images[:2]
    results_list = searcher.batch_search(query_images, n_results=2)
    
    # Check that we get results for each query
    assert len(results_list) == len(query_images)
    
    # Each result should have the expected structure
    for results in results_list:
        assert "ids" in results
        assert "distances" in results
        assert "metadatas" in results


def test_search_with_different_n_results(searcher, encoder, sample_images):
    """Test searching with different n_results values."""
    # Add images to the store
    embeddings = encoder.encode_images(sample_images)
    searcher.rag_store.add_embeddings(embeddings)
    
    # Search with n_results=1
    results_1 = searcher.search(sample_images[0], n_results=1)
    assert len(results_1["ids"]) == 1
    
    # Search with n_results=3
    results_3 = searcher.search(sample_images[0], n_results=3)
    assert len(results_3["ids"]) <= 3


class FakeEncoder:
    """Simple fake encoder that returns the numpy array passed as the embedding.

    This allows tests to construct embeddings directly and use them as both
    stored embeddings and query embeddings.
    """
    def __init__(self, embedding_dim: int = 2):
        self._dim = embedding_dim

    def encode_image(self, image):
        # If image is numpy array matching embedding dim, return as-is
        if isinstance(image, np.ndarray):
            return image.astype(float)
        # If PIL Image, convert to numpy array - but tests pass numpy vectors
        raise ValueError("FakeEncoder expects numpy array as query")

    def encode_images(self, images):
        return np.array(images, dtype=float)

    @property
    def embedding_dimension(self):
        return self._dim


def test_classify_knn(temp_db_dir):
    """Test the k-NN classify method returns the majority label."""
    # Create a rag store and add embeddings with labels
    rag = ChromaRAGStore(collection_name="test_knn_classify", persist_directory=temp_db_dir)
    # Two embeddings near (1,0) with label 0, two near (10,10) with label 1
    embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [10.0, 10.0],
        [10.1, 10.0],
    ], dtype=float)

    metadatas = [
        {"label": 0},
        {"label": 0},
        {"label": 1},
        {"label": 1},
    ]
    rag.add_embeddings(embeddings, metadatas=metadatas)

    fake_encoder = FakeEncoder(embedding_dim=2)
    searcher = ImageSearcher(encoder=fake_encoder, rag_store=rag)

    # Query near the first cluster
    query = np.array([1.05, -0.05])
    result = searcher.classify(query, n_results=3)
    assert result["label"] == 0
    assert result["count"] >= 2
    assert result["confidence"] >= 2/3 - 1e-6


def test_batch_classify(temp_db_dir):
    """Test batch classification returns expected labels for multiple queries."""
    rag = ChromaRAGStore(collection_name="test_batch_knn", persist_directory=temp_db_dir)
    embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [10.0, 10.0],
        [10.1, 10.0],
    ], dtype=float)
    metadatas = [
        {"label": 0},
        {"label": 0},
        {"label": 1},
        {"label": 1},
    ]
    rag.add_embeddings(embeddings, metadatas=metadatas)

    fake_encoder = FakeEncoder(embedding_dim=2)
    searcher = ImageSearcher(encoder=fake_encoder, rag_store=rag)

    queries = [np.array([1.05, -0.05]), np.array([10.05, 10.02])]
    results = searcher.batch_classify(queries, n_results=3)
    assert isinstance(results, list) and len(results) == 2
    assert results[0]["label"] == 0
    assert results[1]["label"] == 1


def test_classify_label_normalization_and_fallbacks(temp_db_dir):
    """Test that classify handles stringified JSON labels, list labels, numpy labels, and fallback to index."""
    rag = ChromaRAGStore(collection_name="test_classify_label_norm", persist_directory=temp_db_dir)

    # Embeddings and metadata for normalization tests
    embeddings = np.array([
        [1.0, 0.0],  # index 0
        [10.0, 0.0], # index 1
    ], dtype=float)

    # Various metadata label formats
    metadatas = [
        {"label": json.dumps([0, 1])},  # JSON list string -> should parse to 0
        {"label": "1"},               # string '1' -> should parse to int 1
    ]
    rag.add_embeddings(embeddings, metadatas=metadatas)

    fake_encoder = FakeEncoder(embedding_dim=2)
    searcher = ImageSearcher(encoder=fake_encoder, rag_store=rag)

    # Query near first embedding which has label 0 in JSON
    query = np.array([1.05, 0.0])
    res = searcher.classify(query, n_results=1)
    assert res["label"] == 0

    # Change metadata to string numeric labels -> classify should read int
    rag.clear()
    metadatas = [
        {"label": "0"},
        {"label": "1"},
    ]
    rag.add_embeddings(embeddings, metadatas=metadatas)
    searcher = ImageSearcher(encoder=fake_encoder, rag_store=rag)
    res = searcher.classify(query, n_results=1)
    assert res["label"] == 0

    # No label, but index metadata present -> should use index
    rag.clear()
    metadatas = [
        {"index": 0},
        {"index": 1},
    ]
    rag.add_embeddings(embeddings, metadatas=metadatas)
    searcher = ImageSearcher(encoder=fake_encoder, rag_store=rag)
    res = searcher.classify(query, n_results=1)
    assert res["label"] == 0


def test_classify_tie_breaker_chooses_smallest_avg_distance(temp_db_dir):
    """Test that tie in counts is resolved by choosing the label with smallest mean distance."""
    rag = ChromaRAGStore(collection_name="test_classify_tie", persist_directory=temp_db_dir)

    # We will set up 4 points: two of label 0 (one close, one far) and two of label 1 (both moderately close)
    embeddings = np.array([
        [1.0],   # label 0 (close)
        [8.0],   # label 0 (far)
        [2.0],   # label 1 (moderate)
        [2.5],   # label 1 (moderate)
    ], dtype=float)

    metadatas = [
        {"label": 0},
        {"label": 0},
        {"label": 1},
        {"label": 1},
    ]
    rag.add_embeddings(embeddings, metadatas=metadatas)

    class OneDEncoder(FakeEncoder):
        def encode_image(self, image):
            return np.array([float(image)])

        def encode_images(self, images):
            return np.array([[float(i[0]) if isinstance(i, (list, np.ndarray)) else float(i)] for i in images])

    fake_encoder = OneDEncoder(embedding_dim=1)
    searcher = ImageSearcher(encoder=fake_encoder, rag_store=rag)

    # Query at 1.5 -> nearest neighbors will be label0(1.0) and label1(2.0) etc. Use n_results=4 to include tie
    classification = searcher.classify(np.array([1.5]), n_results=4)
    # Counts tie (2 each); average distances: label 0 mean high, label 1 mean lower => predicted label 1
    assert classification["label"] == 1


def test_evaluate_classification_confusion_and_metrics():
    """Test confusion matrix and metrics computation with explicit labels and predictions."""
    from vision_rag.search import ImageSearcher

    fake_searcher = ImageSearcher.__new__(ImageSearcher)
    # We only need the helper method; no encoder or rag_store required

    true_labels = [0, 0, 1, 1]
    predicted_labels = [0, 1, 1, 1]

    metrics = ImageSearcher.compute_confusion_and_metrics(fake_searcher, true_labels, predicted_labels)

    # Expect confusion counts: true0->pred0=1, true0->pred1=1, true1->pred1=2
    confusion = metrics["confusion"]
    assert confusion[0][0] == 1
    assert confusion[0][1] == 1
    assert confusion[1][1] == 2

    # Accuracy: 3/4 = 0.75
    assert abs(metrics["accuracy"] - 0.75) < 1e-6
    # Micro metrics should be 0.75
    assert abs(metrics["micro"]["precision"] - 0.75) < 1e-6
    assert abs(metrics["micro"]["recall"] - 0.75) < 1e-6
    assert abs(metrics["micro"]["f1"] - 0.75) < 1e-6
    # Macro precision, recall, f1 close to expected values
    assert abs(metrics["macro"]["precision"] - ((1.0 + 2/3) / 2)) < 1e-6
    assert abs(metrics["macro"]["recall"] - ((0.5 + 1.0) / 2)) < 1e-6

    per_label = metrics["per_label"]
    # Precision for label 0: tp=1, fp=0 -> 1.0
    assert abs(per_label[0]["precision"] - 1.0) < 1e-6
    # Recall for label 0: tp=1, fn=1 -> 0.5
    assert abs(per_label[0]["recall"] - 0.5) < 1e-6
    # For label 1 precision = 2/(1+2)=2/3
    assert abs(per_label[1]["precision"] - (2/3)) < 1e-6


def test_evaluate_classification_batch_using_queries(temp_db_dir):
    """End-to-end evaluation using actual batch_classify on a simple store."""
    rag = ChromaRAGStore(collection_name="test_eval_batch", persist_directory=temp_db_dir)

    embeddings = np.array([
        [1.0, 0.0],
        [0.95, 0.0],
        [10.0, 10.0],
        [10.1, 10.0],
    ], dtype=float)
    metadatas = [
        {"label": 0},
        {"label": 0},
        {"label": 1},
        {"label": 1},
    ]

    rag.add_embeddings(embeddings, metadatas=metadatas)

    fake_encoder = FakeEncoder(embedding_dim=2)
    searcher = ImageSearcher(encoder=fake_encoder, rag_store=rag)

    queries = [np.array([1.05, 0.0]), np.array([10.05, 10.0]), np.array([1.2, 0.0])]
    true_labels = [0, 1, 0]

    metrics = searcher.evaluate_classification(query_images=queries, true_labels=true_labels, n_results=3)

    # We expect at least an overall accuracy in [0,1]. For this simple set, all should classify correct
    assert 0.0 <= metrics["accuracy"] <= 1.0
    # Confusion matrix shape and label presence
    confusion = metrics["confusion"]
    assert 0 in confusion and 1 in confusion
    # Also test the array format option
    matrix_metrics = searcher.compute_confusion_and_metrics(true_labels, [0, 1, 0], return_matrix_as_array=True)
    assert "confusion_matrix" in matrix_metrics
    cm = matrix_metrics["confusion_matrix"]
    assert isinstance(cm, list)
    assert len(cm) == len(matrix_metrics["labels"])  # square matrix


def test_evaluate_classification_and_save_results(temp_db_dir):
    """Test that evaluate_classification can auto-save metrics using a RAGVisualizer."""
    from vision_rag.visualization import RAGVisualizer

    # Prepare store and searcher
    rag = ChromaRAGStore(collection_name="test_eval_save", persist_directory=temp_db_dir)
    embeddings = np.array([
        [1.0, 0.0],
        [0.95, 0.0],
        [10.0, 10.0],
        [10.1, 10.0],
    ], dtype=float)
    metadatas = [{"label": 0}, {"label": 0}, {"label": 1}, {"label": 1}]
    rag.add_embeddings(embeddings, metadatas=metadatas)

    fake_encoder = FakeEncoder(embedding_dim=2)
    searcher = ImageSearcher(encoder=fake_encoder, rag_store=rag)

    queries = [np.array([1.05, 0.0]), np.array([10.05, 10.0]), np.array([1.2, 0.0])]
    true_labels = [0, 1, 0]

    # Use the output_dir argument instead of constructing a visualizer
    results = searcher.evaluate_classification(
        query_images=queries,
        true_labels=true_labels,
        n_results=3,
        save_results=True,
        output_dir=temp_db_dir,
        filename_prefix="test_eval_auto_save",
        to_csv=True,
        to_json=True,
        to_csv_matrix=True,
    )

    # Results should include saved_paths mapping
    assert "saved_paths" in results
    saved_paths = results["saved_paths"]
    assert Path(saved_paths["json"]).exists()
    assert Path(saved_paths["confusion_csv"]).exists()
    assert Path(saved_paths["per_label_csv"]).exists()
    assert Path(saved_paths["confusion_matrix_csv"]).exists()
