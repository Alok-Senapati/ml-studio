"""
Subprocess wrapper utilities for invoking the uv package manager.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def sync_dependencies(project_dir: Path) -> None:
    """
    Install project dependencies using the `uv sync` command.

    Parameters
    ----------
    project_dir : Path
        Target directory containing `pyproject.toml` and `uv.lock`.

    Raises
    ------
    subprocess.CalledProcessError
        If `uv sync` fails with a non-zero exit status code.
    """
    subprocess.run(
        ["uv", "sync"],
        cwd=project_dir,
        check=True,
    )
