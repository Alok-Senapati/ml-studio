"""
Input validation functions for project generation requests.
"""

from __future__ import annotations

import keyword
import re
from pathlib import Path

from .exceptions import (
    InvalidProjectNameError,
    ProjectAlreadyExistsError,
    TemplateNotFoundError,
)

#: Regular expression matching snake_case naming format (e.g. customer_churn)
SNAKE_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_project(
    project_name: str,
    template_dir: Path,
    projects_dir: Path,
) -> None:
    """
    Validate input parameters required to scaffold a new machine learning project.

    Parameters
    ----------
    project_name : str
        Desired name of the project.
    template_dir : Path
        Source template directory path.
    projects_dir : Path
        Target parent projects directory path.

    Raises
    ------
    InvalidProjectNameError
        If `project_name` is empty, not a valid Python identifier, a reserved Python keyword,
        or violates snake_case formatting.
    TemplateNotFoundError
        If `template_dir` path does not exist on disk.
    ProjectAlreadyExistsError
        If target project directory path already exists.
    """
    # Check for empty or whitespace-only project name
    if not project_name.strip():
        raise InvalidProjectNameError("Project name cannot be empty.")

    # Validate Python identifier syntax rules
    if not project_name.isidentifier():
        raise InvalidProjectNameError(f"'{project_name}' is not a valid Python identifier.")

    # Check reserved Python hard and soft keywords
    if keyword.iskeyword(project_name) or keyword.issoftkeyword(project_name):
        raise InvalidProjectNameError(f"'{project_name}' is a reserved Python keyword.")

    # Enforce snake_case naming convention
    if not SNAKE_CASE_PATTERN.fullmatch(project_name):
        raise InvalidProjectNameError("Project name must follow snake_case naming.")

    # Assert source template directory exists
    if not template_dir.exists():
        raise TemplateNotFoundError(f"Template directory not found: {template_dir}")

    # Assert target project path does not already exist
    destination = projects_dir / project_name
    if destination.exists():
        raise ProjectAlreadyExistsError(f"Project '{project_name}' already exists.")
