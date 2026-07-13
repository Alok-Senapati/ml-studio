from __future__ import annotations

import subprocess
from pathlib import Path


def sync_dependencies(project_dir: Path) -> None:
    """Install project dependencies using uv."""

    subprocess.run(
        ["uv", "sync"],
        cwd=project_dir,
        check=True,
    )