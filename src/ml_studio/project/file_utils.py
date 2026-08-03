"""
FileSystem utilities for directory copying, file traversal, and safe text encoding operations.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def copy_directory(source: Path, destination: Path) -> None:
    """
    Recursively copy a directory tree from source to destination path.

    Parameters
    ----------
    source : Path
        Source directory path to copy from.
    destination : Path
        Destination target directory path.

    Raises
    ------
    FileExistsError
        If `destination` directory already exists.
    """
    shutil.copytree(source, destination)


def rename_path(source: Path, destination: Path) -> None:
    """
    Rename a file or directory.

    Parameters
    ----------
    source : Path
        Original path.
    destination : Path
        New target path.
    """
    source.rename(destination)


def find_files(root: Path) -> list[Path]:
    """
    Recursively retrieve all regular file paths under a directory.

    Parameters
    ----------
    root : Path
        Root directory path to search.

    Returns
    -------
    list[Path]
        List of absolute file paths found.
    """
    return [path for path in root.rglob("*") if path.is_file()]


def find_directories(root: Path) -> list[Path]:
    """
    Recursively retrieve all directory paths under a root path.

    Parameters
    ----------
    root : Path
        Root directory path to search.

    Returns
    -------
    list[Path]
        List of absolute directory paths found.
    """
    return [path for path in root.rglob("*") if path.is_dir()]


def is_text_file(path: Path) -> bool:
    """
    Perform a best-effort check to verify if a file can be UTF-8 decoded.

    Parameters
    ----------
    path : Path
        File path to check.

    Returns
    -------
    bool
        True if the file can be decoded as text, False if a UnicodeDecodeError occurs.
    """
    try:
        path.read_text(encoding="utf-8")
        return True
    except UnicodeDecodeError:
        return False


def read_text(path: Path) -> str:
    """
    Read text content from a file using UTF-8 encoding.

    Parameters
    ----------
    path : Path
        File path to read.

    Returns
    -------
    str
        Text content of the file.
    """
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    """
    Write text content to a file using UTF-8 encoding.

    Parameters
    ----------
    path : Path
        File path to write into.
    content : str
        String content to write.
    """
    path.write_text(content, encoding="utf-8")


def delete_directory(path: Path) -> None:
    """
    Remove a directory recursively.

    Parameters
    ----------
    path : Path
        Directory path to delete.
    """
    shutil.rmtree(path)
