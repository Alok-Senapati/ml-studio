from __future__ import annotations

import shutil
from pathlib import Path


def copy_directory(source: Path, destination: Path) -> None:
    """
    Copy a directory recursively.

    Raises:
        FileExistsError:
            If destination already exists.
    """
    shutil.copytree(source, destination)


def rename_path(source: Path, destination: Path) -> None:
    """
    Rename a file or directory.
    """
    source.rename(destination)


def find_files(root: Path) -> list[Path]:
    """
    Return all files under a directory.
    """
    return [path for path in root.rglob("*") if path.is_file()]


def find_directories(root: Path) -> list[Path]:
    """
    Return all directories under a directory.
    """
    return [path for path in root.rglob("*") if path.is_dir()]


def is_text_file(path: Path) -> bool:
    """
    Best-effort check whether a file is text.
    """
    try:
        path.read_text(encoding="utf-8")
        return True
    except UnicodeDecodeError:
        return False


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def delete_directory(path: Path) -> None:
    """
    Remove a directory recursively.
    """
    shutil.rmtree(path)