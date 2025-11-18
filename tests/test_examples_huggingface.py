def test_huggingface_example_imports():
    """Simple smoke test: ensure the huggingface demo module imports and exposes main.

    This test intentionally does not execute the demo (which downloads models)
    — it only checks the module is importable and the `main` function exists.
    """
    import importlib

    mod = importlib.import_module("examples.huggingface_example")
    assert hasattr(mod, "main") and callable(mod.main)
