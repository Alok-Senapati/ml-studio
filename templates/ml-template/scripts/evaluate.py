"""Starter evaluation entry point for the ML project template.

Use this script for offline metrics, validation reports, and plots generated
from a trained model and a held-out dataset split.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the evaluation workflow."""
    parser = argparse.ArgumentParser(description="Run the evaluation pipeline.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/model.pkl"),
        help="Path to the trained model artifact.",
    )
    parser.add_argument(
        "--data-split",
        type=Path,
        default=Path("data/processed/validation.parquet"),
        help="Path to the evaluation dataset split.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("reports/metrics"),
        help="Directory where evaluation outputs should be written.",
    )
    return parser


def main() -> None:
    """Run the placeholder evaluation pipeline."""
    args = build_parser().parse_args()

    print("Evaluation pipeline placeholder")
    print(f"Model path  : {args.model_path}")
    print(f"Data split  : {args.data_split}")
    print(f"Metrics dir : {args.metrics_dir}")
    print("Replace this stub with metric computation, plots, and report generation.")


if __name__ == "__main__":
    main()
