"""
Unit tests for project generation orchestration and cleanup error handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ml_studio.project.generator import generate_project


def test_generate_project(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test standard project scaffolding from template files."""
    template = tmp_path / "template"
    projects = tmp_path / "projects"

    template.mkdir()
    projects.mkdir()

    project_dir = template / "__PROJECT_NAME__"
    project_dir.mkdir()

    readme = project_dir / "README.md"
    readme.write_text("__PROJECT_NAME__\n__PROJECT_DESCRIPTION__", encoding="utf-8")

    monkeypatch.setattr(
        "ml_studio.project.generator.TEMPLATE_DIR",
        template,
    )

    monkeypatch.setattr(
        "ml_studio.project.generator.PROJECTS_DIR",
        projects,
    )

    output = generate_project(
        project_name="customer_churn",
        project_description="Customer Churn Prediction",
    )

    assert output.exists()
    assert output.name == "customer_churn"

    content = (output / "customer_churn" / "README.md").read_text(encoding="utf-8")

    assert "customer_churn" in content


def test_generate_project_cleanup_on_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test automatic cleanup of destination directory if scaffolding fails."""
    template = tmp_path / "template"
    projects = tmp_path / "projects"

    template.mkdir()
    projects.mkdir()

    monkeypatch.setattr(
        "ml_studio.project.generator.TEMPLATE_DIR",
        template,
    )

    monkeypatch.setattr(
        "ml_studio.project.generator.PROJECTS_DIR",
        projects,
    )

    # Mock copy_directory to simulate an unexpected error midway through generation
    def fake_copy_directory(src: Path, dst: Path) -> None:
        dst.mkdir()
        raise RuntimeError("copy failed")

    monkeypatch.setattr("ml_studio.project.generator.copy_directory", fake_copy_directory)

    with pytest.raises(RuntimeError):
        generate_project(project_name="x", project_description="d")

    # Assert that destination directory was cleaned up upon failure
    assert not (projects / "x").exists()


def test_generate_project_sync_calls_sync_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that sync=True triggers dependency installation via uv."""
    template = tmp_path / "template"
    projects = tmp_path / "projects"

    template.mkdir()
    projects.mkdir()

    monkeypatch.setattr(
        "ml_studio.project.generator.TEMPLATE_DIR",
        template,
    )

    monkeypatch.setattr(
        "ml_studio.project.generator.PROJECTS_DIR",
        projects,
    )

    def fake_copy_directory(src: Path, dst: Path) -> None:
        dst.mkdir()

    monkeypatch.setattr("ml_studio.project.generator.copy_directory", fake_copy_directory)

    called: dict[str, Any] = {"sync": False}

    def fake_sync(dest: Path) -> None:
        called["sync"] = True

    monkeypatch.setattr("ml_studio.project.generator.sync_dependencies", fake_sync)

    output = generate_project(project_name="customer_churn", project_description="desc", sync=True)

    assert called["sync"]
    assert output.exists()
