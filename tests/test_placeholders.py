"""
Unit tests for placeholder token substitution in filenames and text contents.
"""

from __future__ import annotations

from pathlib import Path

from ml_studio.project.placeholders import replace_all


def test_replace_all(tmp_path: Path) -> None:
    """Test full placeholder replacement across directory names and text files."""
    project = tmp_path / "__PROJECT_NAME__"
    project.mkdir()

    readme = project / "README.md"
    readme.write_text("__PROJECT_NAME__\n__PROJECT_DESCRIPTION__", encoding="utf-8")

    replace_all(
        tmp_path,
        {
            "__PROJECT_NAME__": "customer_churn",
            "__PROJECT_DESCRIPTION__": "Customer Churn Prediction",
        },
    )

    renamed = tmp_path / "customer_churn"

    assert renamed.exists()

    content = (renamed / "README.md").read_text(encoding="utf-8")

    assert "customer_churn" in content
    assert "Customer Churn Prediction" in content


def test_replace_placeholders_skips_binary_files(tmp_path: Path) -> None:
    """Test that binary files are skipped during content replacement without error."""
    project = tmp_path / "__PROJECT_NAME__"
    project.mkdir()

    binary_file = project / "image.png"
    binary_file.write_bytes(b"\xff\xff\xff")

    placeholders = {"__PROJECT_NAME__": "customer_churn"}

    replace_all(tmp_path, placeholders)

    # Verify binary file was moved with directory rename and content remains unchanged
    new_file = tmp_path / "customer_churn" / "image.png"
    assert new_file.exists()
    assert new_file.read_bytes() == b"\xff\xff\xff"
