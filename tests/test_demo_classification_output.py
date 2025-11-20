import json
import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import demonstrations.demo_with_visualization as demo
from vision_rag.image_store import ImageFileStore as RealImageFileStore


class FakeEncoder:
    """Fast fake encoder that returns deterministic embeddings based on image mean."""
    def __init__(self, embedding_dim: int = 2):
        self._dim = embedding_dim

    def encode_image(self, image):
        arr = np.array(image) if not isinstance(image, np.ndarray) else image
        val = float(arr.mean())
        vec = np.array([val, val * 0.5], dtype=float)
        return vec

    def encode_images(self, images):
        out = []
        for img in images:
            arr = np.array(img) if not isinstance(img, np.ndarray) else img
            val = float(arr.mean())
            out.append([val, val * 0.5])
        return np.array(out)

    @property
    def embedding_dimension(self):
        return self._dim


def fake_load_medmnist_data(dataset_name: str, split: str = "train", root=None):
    # small synthetic dataset: 30 train images, 6 test images
    if split == "train":
        images = np.stack([np.full((28, 28), i * 3 + 10, dtype=np.uint8) for i in range(30)])
        labels = np.array([0 if i < 15 else 1 for i in range(30)], dtype=int)
    else:
        images = np.stack([np.full((28, 28), i * 5 + 20, dtype=np.uint8) for i in range(6)])
        labels = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    return images, labels


def fake_image_file_store(storage_dir: str = None, image_size=None):
    # Use the actual ImageFileStore but with storage dir under temp
    return RealImageFileStore(storage_dir=storage_dir or str(Path(tempfile.gettempdir()) / "image_store_demo"), image_size=image_size)


def test_demo_writes_classification_files(tmp_path, monkeypatch):
    # Prepare temp output directory
    out_dir = tmp_path / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Monkeypatch demo functions/classes
    monkeypatch.setattr(demo, "build_encoder", lambda **kwargs: FakeEncoder())
    monkeypatch.setattr(demo, "load_medmnist_data", fake_load_medmnist_data)
    monkeypatch.setattr(demo, "ImageFileStore", lambda storage_dir="./image_store_demo", image_size=None: fake_image_file_store(str(tmp_path / "image_store_demo"), image_size=image_size))

    class FakeVisualizer(demo.RAGVisualizer):
        def __init__(self, output_dir: str = "./output/visualizations"):
            super().__init__(output_dir=str(out_dir))

    monkeypatch.setattr(demo, "RAGVisualizer", FakeVisualizer)

    # Use a smaller subset to speed up the demo and fewer neighbors
    monkeypatch.setattr(demo, "LARGE_SUBSET", 10)
    monkeypatch.setattr(demo, "NEAREST_NEIGHBORS", 3)

    # Run the demo main function
    demo.main()

    # Verify CSV and JSON files were created
    csv_path = out_dir / "classification_results.csv"
    json_path = out_dir / "classification_results.json"

    assert csv_path.exists(), f"CSV file not created: {csv_path}"
    assert json_path.exists(), f"JSON file not created: {json_path}"

    # Validate CSV contents
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        assert len(rows) > 0
        for r in rows:
            assert "predicted_label" in r
            assert "confidence" in r
            assert "correct" in r

    # Validate JSON contents
    with open(json_path, "r", encoding="utf-8") as jf:
        data = json.load(jf)
        assert isinstance(data, list)
        assert len(data) == len(rows)
        for item in data:
            assert "predicted_label" in item
            assert "confidence" in item
            assert "correct" in item
