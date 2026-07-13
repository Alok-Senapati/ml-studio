from __future__ import annotations

import argparse
from pathlib import Path

from .generator import generate_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-studio",
        description="Generate a new ML project.",
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Project name (snake_case)",
    )

    parser.add_argument(
        "--description",
        required=True,
        help="Project description",
    )

    parser.add_argument(
        "--author",
        default="Alok Senapati",
    )

    parser.add_argument(
        "--year",
        default="2026",
    )

    parser.add_argument(
        "--sync",
        action="store_true",
        help="Install dependencies after project creation.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    project: Path = generate_project(
        project_name=args.name,
        project_description=args.description,
        author=args.author,
        year=args.year,
        sync=args.sync
    )

    print()
    print("========================================")
    print(" Project created successfully")
    print("========================================")
    print(f" Name : {args.name}")
    print(f" Path : {project}")
    print("========================================")


if __name__ == "__main__":
    main()