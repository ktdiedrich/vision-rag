"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path
import pytest

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Mark for tests that require network access
network_required = pytest.mark.skipif(
    True,  # Skip by default since Zenodo is unreliable
    reason="Requires network access to Zenodo which may be unavailable"
)
