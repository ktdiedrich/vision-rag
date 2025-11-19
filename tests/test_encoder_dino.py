import numpy as np
import pytest

from vision_rag.encoder import DINOImageEncoder


def test_dino_encoder_encode_image_and_images():
    """Test DINOImageEncoder encodes images and matches embedding_dimension.

    This test will be skipped if the DINO model can't be downloaded from
    Hugging Face (e.g. on offline CI). We intentionally skip to keep CI
    environments that block large downloads operational.
    """
    try:
        encoder = DINOImageEncoder()
    except Exception as e:
        pytest.skip(f"Skipping DINO encoder tests as model cannot be loaded: {e}")

    # Single image: grayscale dummy
    img = np.zeros((28, 28), dtype=np.uint8)
    emb = encoder.encode_image(img)

    # Check dimension
    assert emb.ndim == 1
    assert emb.shape[0] == encoder.embedding_dimension

    # Multiple images
    imgs = [img, np.ones((28, 28), dtype=np.uint8) * 128]
    embs = encoder.encode_images(imgs)
    assert embs.ndim == 2
    assert embs.shape == (len(imgs), encoder.embedding_dimension)


def test_dino_embedding_dimension_property():
    try:
        encoder = DINOImageEncoder()
    except Exception as e:
        pytest.skip(f"Skipping DINO encoder tests as model cannot be loaded: {e}")

    # embedding_dimension should be integer > 0
    dim = encoder.embedding_dimension
    assert isinstance(dim, int) and dim > 0
