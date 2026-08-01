from pathlib import Path

from ml_studio.project.generator import generate_project


def test_generate_project(monkeypatch, tmp_path: Path) -> None:
    template = tmp_path / "template"
    projects = tmp_path / "projects"

    template.mkdir()
    projects.mkdir()

    project_dir = template / "__PROJECT_NAME__"
    project_dir.mkdir()

    readme = project_dir / "README.md"
    readme.write_text("__PROJECT_NAME__\n__PROJECT_DESCRIPTION__")

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

    content = (output / "customer_churn" / "README.md").read_text()

    assert "customer_churn" in content
