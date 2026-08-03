"""
Unit tests for uv subprocess invocation helper routines.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from ml_studio.project.uv_utils import sync_dependencies


def test_sync_dependencies_calls_subprocess_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that sync_dependencies invokes subprocess.run with ['uv', 'sync']."""
    called: dict[str, Any] = {"args": None}

    def fake_run(args: list[str], cwd: Path, check: bool) -> None:
        called["args"] = args
        called["cwd"] = cwd
        called["check"] = check

    monkeypatch.setattr(subprocess, "run", fake_run)

    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    sync_dependencies(project_dir)

    assert called["args"] == ["uv", "sync"]
    assert called["cwd"] == project_dir
    assert called["check"] is True
