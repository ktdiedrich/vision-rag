# Import Guidelines for vision-rag

This document provides a short, specific rule to follow for imports in the `vision-rag` package. It is meant to complement the existing `CONTRIBUTING.md` rules and to guide GitHub Copilot suggestions so generated code follows the project style.

## Rule: place imports at top-of-file

- All top-level/standard/non-optional imports should appear at the top of the file. This helps with readability, avoids hidden side effects, and makes static analysis (type checking, linters, import sorting) reliable.
- Top-of-file means immediately after module-level docstring and copyright/license, before any class or function definitions.

Example (preferred):

```python
"""Module docstring (if present).
"""

from typing import List, Optional
import numpy as np
from PIL import Image

# Local imports
from vision_rag.config import CLIP_MODEL_NAME

# rest of file follows
```

## Exceptions to the rule

Sometimes inline imports are acceptable — document the reason in a comment and keep them minimal:

- Avoid circular imports: if importing at top introduces a circular import, import locally inside the function and add a comment explaining why.
- Optional/slow dependencies: if a dependency is optional or heavy (e.g., `transformers`), you may import it inside a function or class initialization to avoid startup overhead and to allow environments without the dependency to still import the module.
- Tests or dev-only code may occasionally import locally to minimize load time.

Example (optional dependency):

```python
# Only import the optional model if requested by the user to avoid heavy downloads
try:
    from transformers import AutoModel, AutoImageProcessor
except Exception:  # pragma: no cover - optional dependency environment
    AutoModel = None

class DINOImageEncoder:
    def __init__(self, ...):
        if AutoModel is None:
            raise RuntimeError("transformers package is required for DINOImageEncoder")
```

## GitHub Copilot / AI assistants usage

When using GitHub Copilot, add a short directive comment if you want the assistant to generate code based on existing imports or to add imports in the correct place.

- Add an instruction at the top of the file (or near the top) before writing the code you want Copilot to complete:

```python
# COPILOT: Add any required imports at the top of this file, not inline inside functions.
```

- If Copilot suggests inline imports as part of a function body, you can prompt it with: `# Move imports to the top of the file`.
- Always review generated import statements. If they are new top-level imports, edit the top of the file to include them.

## Linting & enforcement

- We recommend running `isort` and `ruff` in CI to ensure imports are ordered and style is enforced.
- Add the right configuration to `pyproject.toml` (if not present):

```toml
[tool.isort]
profile = "black"

[tool.ruff]
select = ["I"]  # enable import-related linting
```

This file is part of `vision-rag` documentation and is intended to be a short, easy-to-follow directive that developers and AI assistants can follow.
---
