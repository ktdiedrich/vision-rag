import numpy as np
import pytest
from transformers import AutoImageProcessor, AutoModel

def test_hf_dino_import_and_forward():
    """Smoke test: load facebook/dino-vits8 feature extractor + model and run a forward.

    HuggingFace DION models at https://huggingface.co/collections/facebook/dinov3 
    This test will be skipped if the model cannot be downloaded (e.g., no
    network in CI). We intentionally skip rather than fail to avoid
    blocking environments that do not allow large downloads.
    """
    model_name = "facebook/dino-vits8"

    try:
        feat = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        # Skip if we cannot access Hugging Face resources from the environment
        pytest.skip(f"Skipping DINO model test as it cannot be loaded: {e}")

    # Prepare a fake image tensor and run a forward pass
    img = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
    inputs = feat(images=[img], return_tensors="pt")

    # Run forward and ensure the expected output exists
    outputs = model(**inputs)
    assert hasattr(outputs, "last_hidden_state")
    # Ensure output's last_hidden_state has expected number of dims (batch, seq, hidden)
    assert outputs.last_hidden_state.ndim == 3
