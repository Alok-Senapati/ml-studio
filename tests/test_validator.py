"""
Unit tests for input project name and directory validation logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_studio.project.exceptions import (
    InvalidProjectNameError,
    ProjectAlreadyExistsError,
    TemplateNotFoundError,
)
from ml_studio.project.validator import validate_project


def test_valid_project(tmp_path: Path) -> None:
    """Test that a valid snake_case project name passes validation."""
    template = tmp_path / "template"
    projects = tmp_path / "projects"

    template.mkdir()
    projects.mkdir()

    validate_project("customer_churn", template, projects)


def test_empty_project_name(tmp_path: Path) -> None:
    """Test that an empty string project name raises InvalidProjectNameError."""
    template = tmp_path / "template"
    projects = tmp_path / "projects"

    template.mkdir()
    projects.mkdir()

    with pytest.raises(InvalidProjectNameError):
        validate_project("", template, projects)


@pytest.mark.parametrize(
    "name",
    [
        "customer-churn",
        "customer churn",
        "123project",
        "customer.churn",
    ],
)
def test_invalid_identifier(
    name: str,
    tmp_path: Path,
) -> None:
    """Test that invalid Python identifiers raise InvalidProjectNameError."""
    template = tmp_path / "template"
    projects = tmp_path / "projects"

    template.mkdir()
    projects.mkdir()

    with pytest.raises(InvalidProjectNameError):
        validate_project(name, template, projects)


@pytest.mark.parametrize(
    "name",
    ["class", "for", "while", "match", "import", "case"],
)
def test_python_keywords(
    name: str,
    tmp_path: Path,
) -> None:
    """Test that reserved Python keywords raise InvalidProjectNameError."""
    template = tmp_path / "template"
    projects = tmp_path / "projects"

    template.mkdir()
    projects.mkdir()

    with pytest.raises(InvalidProjectNameError):
        validate_project(name, template, projects)


@pytest.mark.parametrize(
    "name",
    [
        "CustomerChurn",
        "customerChurn",
        "CUSTOMER_CHURN",
    ],
)
def test_snake_case_validation(
    name: str,
    tmp_path: Path,
) -> None:
    """Test that non-snake_case names raise InvalidProjectNameError."""
    template = tmp_path / "template"
    projects = tmp_path / "projects"

    template.mkdir()
    projects.mkdir()

    with pytest.raises(InvalidProjectNameError):
        validate_project(name, template, projects)


def test_existing_project(tmp_path: Path) -> None:
    """Test that an existing destination project directory raises ProjectAlreadyExistsError."""
    template = tmp_path / "template"
    projects = tmp_path / "projects"

    template.mkdir()
    projects.mkdir()

    (projects / "customer_churn").mkdir()

    with pytest.raises(ProjectAlreadyExistsError):
        validate_project("customer_churn", template, projects)


def test_missing_template(tmp_path: Path) -> None:
    """Test that a non-existent template directory raises TemplateNotFoundError."""
    projects = tmp_path / "projects"
    projects.mkdir()

    with pytest.raises(TemplateNotFoundError):
        validate_project(
            "customer_churn",
            tmp_path / "missing_template",
            projects,
        )
