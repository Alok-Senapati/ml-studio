from __future__ import annotations

import argparse
from pathlib import Path

from .generator import generate_project
from .exceptions import ProjectGeneratorError
from .logger import logger


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

    try:
        logger.info("Creating project...")

        project = generate_project(
            project_name=args.name,
            project_description=args.description,
            author=args.author,
            year=args.year,
            sync=args.sync,
        )

    except ProjectGeneratorError as exc:
        logger.error("")
        logger.error("Project creation failed")
        logger.error(str(exc))
        raise SystemExit(1)

    logger.info("")
    logger.info("Project created successfully")
    logger.info("----------------------------------------")
    logger.info(f"Name : {args.name}")
    logger.info(f"Path : {project}")
    logger.info("----------------------------------------")


if __name__ == "__main__":
    main()
