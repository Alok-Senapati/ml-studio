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


def test_generate_project_cleanup_on_error(monkeypatch, tmp_path: Path) -> None:
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

    # make copy_directory raise after creating destination to simulate error
    def fake_copy_directory(src, dst):
        dst.mkdir()
        raise RuntimeError("copy failed")

    monkeypatch.setattr("ml_studio.project.generator.copy_directory", fake_copy_directory)

    with pytest.raises(RuntimeError):
        generate_project(project_name="x", project_description="d")

    # destination should be removed by cleanup
    assert not (projects / "x").exists()


def test_generate_project_sync_calls_sync_dependencies(monkeypatch, tmp_path: Path) -> None:
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

    # fake copy_directory that creates destination
    def fake_copy_directory(src, dst):
        dst.mkdir()

    monkeypatch.setattr("ml_studio.project.generator.copy_directory", fake_copy_directory)

    called = {"sync": False}

    def fake_sync(dest):
        called["sync"] = True

    monkeypatch.setattr("ml_studio.project.generator.sync_dependencies", fake_sync)

    output = generate_project(project_name="customer_churn", project_description="desc", sync=True)

    assert called["sync"]
    assert output.exists()
