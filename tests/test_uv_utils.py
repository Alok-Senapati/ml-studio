from pathlib import Path
import subprocess

from ml_studio.project.uv_utils import sync_dependencies


def test_sync_dependencies_calls_subprocess_run(monkeypatch, tmp_path: Path) -> None:
    called = {"args": None}

    def fake_run(args, cwd, check):
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
