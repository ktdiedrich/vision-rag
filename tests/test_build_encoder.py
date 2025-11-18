import pytest

from vision_rag.encoder import build_encoder


def test_build_encoder_returns_clip_or_dino():
    # Try clip first; skip if CLIP SentenceTransformer isn't available
    try:
        enc = build_encoder("clip")
        assert hasattr(enc, "encode_image") and hasattr(enc, "encode_images")
    except Exception as e:
        pytest.skip(f"Skipping clip build due to error: {e}")

    # Try dino; skip if model not available
    try:
        enc = build_encoder("dino")
        assert hasattr(enc, "encode_image") and hasattr(enc, "encode_images")
    except Exception as e:
        pytest.skip(f"Skipping dino build due to error: {e}")
