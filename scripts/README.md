# Helper scripts

This folder contains small helper scripts used for running demos and preparing the test environment.

download_test_datasets.py
- Purpose: pre-download MedMNIST datasets for local development or CI.
- Default behaviour: downloads a set of datasets (OrganSMNIST, PathMNIST, PneumoniaMNIST) at size=224.
- Usage examples:

```bash
# Dry-run (doesn't download):
python scripts/download_test_datasets.py --dry-run

# Download a list of datasets to the repo `data/` folder in small size:
python scripts/download_test_datasets.py --datasets OrganSMNIST PathMNIST --size 224 --root ./data
```

Tip: Use size=28 or size=64 in CI to avoid very large downloads (size=224 files can be several gigabytes).
