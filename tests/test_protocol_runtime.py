import pytest

from vision_rag.encoder import ImageEncoderProtocol, CLIPImageEncoder, DINOImageEncoder


def test_protocol_runtime_checkable_clip():
    try:
        enc = CLIPImageEncoder()
    except Exception as e:
        pytest.skip(f"Skip: could not initialize CLIP encoder: {e}")

    assert isinstance(enc, ImageEncoderProtocol)


def test_protocol_runtime_checkable_dino():
    try:
        enc = DINOImageEncoder()
    except Exception as e:
        pytest.skip(f"Skip: could not initialize DINO encoder: {e}")

    assert isinstance(enc, ImageEncoderProtocol)
