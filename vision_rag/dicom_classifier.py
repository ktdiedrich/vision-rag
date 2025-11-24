"""DICOMClassifier: fine-tune a DINO (ViT) model for image classification using HuggingFace Trainer.

This module provides a small wrapper around HuggingFace's `Trainer` to train
an `AutoModelForImageClassification` model (e.g., facebook/dino-vits8) on a set
of images and labels. It uses the model's `AutoImageProcessor` for input
processing and implements a lightweight torch Dataset wrapper for training.

Note: This is purposely simple for demo/training on small subsets. For
production training use the full HuggingFace `datasets` and distributed
training strategies described in the Transformers documentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class _DICOMDataset(Dataset):
    """A minimal torch Dataset that uses an AutoImageProcessor to prepare inputs.

    Images are expected to be either PIL images or numpy arrays; labels are ints.
    The dataset applies the processor in __getitem__ so that it is compatible
    with `Trainer` and Torch DataLoader collators.
    """

    def __init__(self, images: List[Any], labels: List[int], processor: AutoImageProcessor):
        assert len(images) == len(labels), "images and labels length mismatch"
        self.images = images
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        # Convert numpy arrays to PIL for processor convenience
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
            elif img.shape[-1] == 1:
                img = Image.fromarray(img.squeeze().astype(np.uint8)).convert("RGB")
            else:
                img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
        elif not isinstance(img, Image.Image):
            # Attempt to convert
            img = Image.fromarray(np.asarray(img).astype(np.uint8)).convert("RGB")
        else:
            # If it's already a PIL Image, ensure it's RGB (avoid ImageProcessor errors)
            if img.mode != "RGB":
                img = img.convert("RGB")

        # Prepare inputs (pixel_values) using the image processor
        enc = self.processor(images=img, return_tensors="pt")
        # enc is a dict with 'pixel_values' shaped (1, C, H, W); we squeeze to remove batch dim
        sample = {k: v.squeeze(0) for k, v in enc.items()}
        sample["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample


class DICOMClassifier:
    """Wrapper class to fine-tune a DINO (ViT) model for image classification.

    Example usage:
      classifier = DICOMClassifier(model_name='facebook/dino-vits8', num_labels=4)
      trainer = classifier.fit(train_images, train_labels, val_images, val_labels, output_dir='./dino_class')

    After training you can reload the fine-tuned model (the checkpoint in output_dir)
    and use `DINOImageEncoder` with that checkpoint to produce embeddings from the
    fine-tuned backbone.
    """

    def __init__(self, model_name: str = "facebook/dino-vits8", num_labels: int = 2, device: Optional[str] = None):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        # Prepare model with classification head
        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label={i: str(i) for i in range(num_labels)},
            label2id={str(i): i for i in range(num_labels)},
        )
        self.model.to(self.device)

    def _compute_metrics(self):
        def compute_metric(pred):
            labels = pred.label_ids
            preds = np.argmax(pred.predictions, axis=1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
            return {
                "accuracy": accuracy_score(labels, preds),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        return compute_metric

    def fit(
        self,
        train_images: List[Any],
        train_labels: List[int],
        val_images: Optional[List[Any]] = None,
        val_labels: Optional[List[int]] = None,
        output_dir: str = "./dino_classifier",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        learning_rate: float = 5e-5,
        logging_steps: int = 10,
        save_strategy: str = "epoch",
        evaluation_strategy: str = "epoch",
        **kwargs,
    ) -> Tuple[Trainer, str]:
        """Train the classifier using transformers Trainer.

        Returns the Trainer and the output_dir where the model is saved.
        """
        # Build datasets
        train_dataset = _DICOMDataset(train_images, train_labels, self.processor)
        eval_dataset = None
        if val_images is not None and val_labels is not None:
            eval_dataset = _DICOMDataset(val_images, val_labels, self.processor)

        # Training arguments
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # Try to be flexible with Transformers version: some versions accept
        # evaluation_strategy/save_strategy kwargs, others don't. Attempt the
        # more featureful constructor first, and fall back to a minimal one
        # if the current installed `transformers` doesn't support those args.
        try:
            training_args = TrainingArguments(
                output_dir=str(output_path),
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                evaluation_strategy=evaluation_strategy,
                save_strategy=save_strategy,
                learning_rate=learning_rate,
                logging_steps=logging_steps,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                push_to_hub=False,
                **kwargs,
            )
        except TypeError:
            # Fall back to a conservative set of args for older transformers
            training_args = TrainingArguments(
                output_dir=str(output_path),
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                learning_rate=learning_rate,
                logging_steps=logging_steps,
                push_to_hub=False,
                **kwargs,
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.processor,
            data_collator=default_data_collator,
            compute_metrics=self._compute_metrics(),
        )

        trainer.train()
        trainer.save_model(str(output_path))
        return trainer, str(output_path)


__all__ = ["DICOMClassifier"]
