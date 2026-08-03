from pathlib import Path

from ml_studio.project.placeholders import replace_all


def test_replace_all(tmp_path: Path) -> None:
    project = tmp_path / "__PROJECT_NAME__"
    project.mkdir()

    readme = project / "README.md"
    readme.write_text("__PROJECT_NAME__\n__PROJECT_DESCRIPTION__")

    replace_all(
        tmp_path,
        {
            "__PROJECT_NAME__": "customer_churn",
            "__PROJECT_DESCRIPTION__": "Customer Churn Prediction",
        },
    )

    renamed = tmp_path / "customer_churn"

    assert renamed.exists()

    content = (renamed / "README.md").read_text()

    assert "customer_churn" in content
    assert "Customer Churn Prediction" in content


def test_replace_placeholders_skips_binary_files(tmp_path: Path) -> None:
    # create a binary file that should be skipped by replace_placeholders_in_content
    project = tmp_path / "__PROJECT_NAME__"
    project.mkdir()

    binary_file = project / "image.png"
    binary_file.write_bytes(b"\xff\xff\xff")

    placeholders = {"__PROJECT_NAME__": "customer_churn"}

    # should not raise and binary file content should remain unchanged
    replace_all(tmp_path, placeholders)

    # the directory will be renamed, so check the new location
    new_file = tmp_path / "customer_churn" / "image.png"
    assert new_file.exists()
    assert new_file.read_bytes() == b"\xff\xff\xff"
