from __future__ import annotations

import keyword
import re
from pathlib import Path

from .exceptions import (
    InvalidProjectNameError,
    ProjectAlreadyExistsError,
    TemplateNotFoundError,
)

SNAKE_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_project(
    project_name: str,
    template_dir: Path,
    projects_dir: Path,
) -> None:
    """
    Validate inputs required to create a new ML project.

    Raises:
        InvalidProjectNameError
        ProjectAlreadyExistsError
        TemplateNotFoundError
    """

    if not project_name.strip():
        raise InvalidProjectNameError("Project name cannot be empty.")

    if not project_name.isidentifier():
        raise InvalidProjectNameError(f"'{project_name}' is not a valid Python identifier.")

    if keyword.iskeyword(project_name) or keyword.issoftkeyword(project_name):
        raise InvalidProjectNameError(f"'{project_name}' is a reserved Python keyword.")

    if not SNAKE_CASE_PATTERN.fullmatch(project_name):
        raise InvalidProjectNameError("Project name must follow snake_case naming.")

    if not template_dir.exists():
        raise TemplateNotFoundError(f"Template directory not found: {template_dir}")

    destination = projects_dir / project_name

    if destination.exists():
        raise ProjectAlreadyExistsError(f"Project '{project_name}' already exists.")
