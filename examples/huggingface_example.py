"""
Simple Hugging Face example — loads a small dataset and runs a transformer model.

This example uses `datasets` to load a small subset of `ag_news` and
`transformers` to tokenize and encode a small batch of texts. It also
demonstrates using a pipeline for sentiment classification (SST-2).

Try it with:
    python examples/huggingface_example.py

This is intentionally small and uses short splits (e.g. `[:10]`) to
keep downloads minimal for example purposes.
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, pipeline


def main():
    print("\n== Hugging Face Example: datasets + transformers ==\n")

    # 1. Load small subset of a text dataset (ag_news is small and common)
    print("Loading sample dataset (ag_news, 10 examples)...")
    ds = load_dataset("ag_news", split="train[:10]")

    print(f"Loaded {len(ds)} examples from 'ag_news' (train[:10])")
    # Helper for robust text extraction from a dataset example
    def _get_text_from_example(ex):
        if isinstance(ex, dict):
            # Common text field names across datasets
            for key in ("text", "sentence", "content", "article", "body", "title"):
                if key in ex:
                    return ex[key]
            # Fallback to string conversion of the example
            return str(ex)
        # If the example is already a string
        return str(ex)

    print("Example texts:")
    for i, ex in enumerate(ds[:3]):
        text = _get_text_from_example(ex)
        print(f" {i+1}. {text[:120]}...")

    # 2. Tokenize and embed with a transformer model (fast base model)
    model_name = "distilbert-base-uncased"
    print(f"\nLoading tokenizer + model: {model_name} (this may download files)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    sample_texts = [_get_text_from_example(ex) for ex in ds[:3]]
    print("Tokenizing a small batch (3 texts)")
    tokens = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt")

    print("Running model to get embeddings (may be slow on CPU)...")
    outputs = model(**tokens)
    # outputs.last_hidden_state shape: (batch, seq_len, hidden_size)
    # We'll use mean pooling of the hidden states as a simple embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    print(f"Embeddings shape: {embeddings.shape}")

    # Show a sample vector (first 5 dims)
    print("Sample embedding (first 5 dims):", embeddings[0][:5].tolist())

    # 3. Use a fast pipeline for text classification (finetuned on SST-2)
    clf_model = "distilbert-base-uncased-finetuned-sst-2-english"
    print(f"\nLoading classification pipeline ({clf_model})")
    sentiment = pipeline("sentiment-analysis", model=clf_model)

    for i, text in enumerate(sample_texts):
        print(f"\nText {i+1}: {text[:120]}...")
        pred = sentiment(text[:512])[0]
        print(f"  -> Label: {pred['label']}, score={pred['score']:.3f}")

    print("\nDemo finished — remove large downloads with `datasets`/`transformers` cache cleanup if desired.")


if __name__ == "__main__":
    main()
