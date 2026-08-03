"""
Project generator orchestrator combining template copy, placeholder replacement, and sync.
"""

from __future__ import annotations

from pathlib import Path

from .constants import PROJECTS_DIR, TEMPLATE_DIR
from .file_utils import copy_directory, delete_directory
from .logger import logger
from .placeholders import replace_all
from .uv_utils import sync_dependencies
from .validator import validate_project


def generate_project(
    project_name: str,
    project_description: str,
    author: str = "Alok Senapati",
    year: str = "2026",
    sync: bool = False,
) -> Path:
    """
    Generate a new ML project scaffolded from template files.

    Parameters
    ----------
    project_name : str
        Name of the project in `snake_case`.
    project_description : str
        Short text description summarizing the project objective.
    author : str, default="Alok Senapati"
        Name of the project author for metadata configuration.
    year : str, default="2026"
        Creation year string for license headers.
    sync : bool, default=False
        If True, run `uv sync` to install dependencies upon creation.

    Returns
    -------
    Path
        Path to the generated project directory.

    Raises
    ------
    InvalidProjectNameError
        If `project_name` fails snake_case, keyword, or identifier validation.
    ProjectAlreadyExistsError
        If destination project directory already exists.
    TemplateNotFoundError
        If the template directory cannot be found.
    """
    logger.info("Validating project...")
    validate_project(
        project_name=project_name,
        template_dir=TEMPLATE_DIR,
        projects_dir=PROJECTS_DIR,
    )

    logger.info("Copying template...")
    destination = PROJECTS_DIR / project_name

    placeholders = {
        "__PROJECT_NAME__": project_name,
        "__PROJECT_DESCRIPTION__": project_description,
        "__AUTHOR__": author,
        "__YEAR__": year,
    }

    try:
        # Copy raw template directory structure to destination
        copy_directory(TEMPLATE_DIR, destination)
        logger.info("Replacing placeholders...")

        # Substitute template placeholder strings in file paths and text content
        replace_all(destination, placeholders)
    except Exception:
        # Cleanup destination directory if generation encounters an error
        if destination.exists():
            delete_directory(destination)
        raise

    # Optionally sync dependencies via uv
    if sync:
        logger.info("Installing dependencies...")
        sync_dependencies(destination)

    return destination
