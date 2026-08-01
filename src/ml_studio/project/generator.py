from __future__ import annotations

from pathlib import Path

from .constants import PROJECTS_DIR, TEMPLATE_DIR
from .file_utils import copy_directory, delete_directory
from .placeholders import replace_all
from .validator import validate_project
from .uv_utils import sync_dependencies
from .logger import logger


def generate_project(
    project_name: str,
    project_description: str,
    author: str = "Alok Senapati",
    year: str = "2026",
    sync: bool = False,
) -> Path:
    """
    Generate a new ML project from the template.

    Returns
    -------
    Path
        Path to the generated project.
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
        copy_directory(TEMPLATE_DIR, destination)
        logger.info("Replacing placeholders...")
        replace_all(destination, placeholders)
    except Exception:
        if destination.exists():
            delete_directory(destination)
        raise

    if sync:
        logger.info("Installing dependencies...")
        sync_dependencies(destination)

    return destination
