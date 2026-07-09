"""Starter training entry point for the ML project template.

The implementation is intentionally lightweight. Keep the CLI stable while you
replace the placeholder body with project-specific data loading, training, and
artifact persistence.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the training workflow."""
    parser = argparse.ArgumentParser(description="Run the training pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to the training configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory where trained artifacts should be written.",
    )
    return parser


def main() -> None:
    """Run the placeholder training pipeline."""
    args = build_parser().parse_args()

    print("Training pipeline placeholder")
    print(f"Config file : {args.config}")
    print(f"Output dir  : {args.output_dir}")
    print("Replace this stub with dataset loading, model fitting, and model export.")


if __name__ == "__main__":
    main()
