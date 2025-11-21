#!/usr/bin/env python3
"""Script to replot evaluation metrics using current visualizer code.

Usage:
    python replot_eval_plots.py [metrics_json_path] [output_dir]

Loads an evaluation JSON (like 07_eval_large.json) and regenerates
confusion matrix and per-label metrics plots using RAGVisualizer.
"""
import sys
from pathlib import Path
import json
# Ensure repository root is in sys.path for importing vision_rag
sys.path.insert(0, str(Path(__file__).parent.parent))
from vision_rag import RAGVisualizer


def main():
    metrics_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('output/visualizations/07_eval_large.json')
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('output/visualizations')

    if not metrics_path.exists():
        print(f"Metrics file {metrics_path} not found.")
        raise SystemExit(1)

    with open(metrics_path, 'r') as fh:
        metrics = json.load(fh)

    viz = RAGVisualizer(output_dir=str(output_dir))

    # regenerate plots
    cm_png = viz.plot_confusion_matrix(metrics['confusion'], labels=metrics.get('labels'), filename='07_confusion_matrix_large.png')
    per_png = viz.plot_per_label_metrics(metrics['per_label'], labels_order=metrics.get('labels'), filename='07_per_label_metrics_large.png')

    print('Regenerated:')
    print('  -', cm_png)
    print('  -', per_png)


if __name__ == '__main__':
    main()
