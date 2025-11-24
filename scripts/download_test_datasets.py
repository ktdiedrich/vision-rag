#!/usr/bin/env python3
"""Small helper to pre-download MedMNIST datasets for tests/CI.

Usage: python scripts/download_test_datasets.py --datasets OrganSMNIST PathMNIST --size 28 --root ./data

This script intentionally defaults to small sizes (28) for CI-friendly downloads.
It supports a dry-run mode (prints what it would download) so CI can decide whether
to fetch large files.

Author: automation (added by dev assistant)
"""
from __future__ import annotations

import argparse
import sys
from typing import Iterable

from vision_rag.data_loader import download_medmnist


DEFAULT_DATASETS = [
    "OrganSMNIST",
    "PathMNIST",
    "PneumoniaMNIST",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-download MedMNIST datasets for test/CI use")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Dataset names to download (default: OrganSMNIST PathMNIST PneumoniaMNIST)")
    p.add_argument("--size", type=int, default=224, choices=[28, 64, 128, 224], help="Image size to download (default: 224)")
    p.add_argument("--root", type=str, default="./data", help="Root directory to save dataset files (default: ./data)")
    p.add_argument("--dry-run", action="store_true", help="Print what would be downloaded without performing network requests")
    return p.parse_args(argv)


def validate_datasets(names: Iterable[str]) -> list[str]:
    # Minimal validation: ensure non-empty strings
    validated = [n.strip() for n in names if n and n.strip()]
    if not validated:
        raise ValueError("No dataset names provided")
    return validated


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    datasets = validate_datasets(args.datasets)

    print(f"Target root: {args.root}")
    print(f"Size: {args.size}")
    print("Datasets:")
    for d in datasets:
        print(f"  - {d}")

    if args.dry_run:
        print("\nDry-run mode enabled — not downloading. Exit 0.")
        return 0

    # Download each dataset (skip when already present)
    for d in datasets:
        try:
            print(f"\nDownloading {d} (size={args.size}) to {args.root} ...")
            download_medmnist(dataset_name=d, root=args.root, size=args.size)
        except Exception as e:  # keep downloads robust and continue on errors
            print(f"Failed to download {d}: {e}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
