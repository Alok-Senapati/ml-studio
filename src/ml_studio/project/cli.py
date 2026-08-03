"""
Command Line Interface (CLI) entry point for ml-studio project generation.
"""

from __future__ import annotations

import argparse

from .exceptions import ProjectGeneratorError
from .generator import generate_project
from .logger import logger


def build_parser() -> argparse.ArgumentParser:
    """
    Construct the command line argument parser for `ml-studio`.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser instance with all CLI options.
    """
    parser = argparse.ArgumentParser(
        prog="ml-studio",
        description="Generate a new production-ready ML project template.",
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Project name in snake_case format.",
    )

    parser.add_argument(
        "--description",
        required=True,
        help="Brief description of the machine learning project.",
    )

    parser.add_argument(
        "--author",
        default="Alok Senapati",
        help="Author name for package metadata.",
    )

    parser.add_argument(
        "--year",
        default="2026",
        help="Copyright year for license header.",
    )

    parser.add_argument(
        "--sync",
        action="store_true",
        help="Automatically install project dependencies after creation using uv.",
    )

    return parser


def main() -> None:
    """
    Main CLI entry point function executing project generator command.

    Raises
    ------
    SystemExit
        Exits with code 1 if project creation fails due to validation errors.
    """
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
        raise SystemExit(1) from exc

    logger.info("")
    logger.info("Project created successfully")
    logger.info("----------------------------------------")
    logger.info(f"Name : {args.name}")
    logger.info(f"Path : {project}")
    logger.info("----------------------------------------")


if __name__ == "__main__":
    main()
