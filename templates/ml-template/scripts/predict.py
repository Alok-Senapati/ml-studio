"""Starter batch prediction entry point for the ML project template.

Use this workflow to load a trained model, score input records, and persist the
resulting predictions in a project-specific format.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the prediction workflow."""
    parser = argparse.ArgumentParser(description="Run the prediction pipeline.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/model.pkl"),
        help="Path to the trained model artifact.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/predict.parquet"),
        help="Path to the batch input data.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("reports/predictions.parquet"),
        help="Path where predictions should be written.",
    )
    return parser


def main() -> None:
    """Run the placeholder prediction pipeline."""
    args = build_parser().parse_args()

    print("Prediction pipeline placeholder")
    print(f"Model path  : {args.model_path}")
    print(f"Input path  : {args.input_path}")
    print(f"Output path : {args.output_path}")
    print("Replace this stub with feature loading, scoring, and result persistence.")


if __name__ == "__main__":
    main()
