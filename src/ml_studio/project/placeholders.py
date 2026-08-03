"""
Placeholder substitution logic for paths and file text content during scaffolding.
"""

from __future__ import annotations

from pathlib import Path

from .file_utils import (
    find_directories,
    find_files,
    is_text_file,
    read_text,
    rename_path,
    write_text,
)


def replace_placeholders_in_content(
    root: Path,
    placeholders: dict[str, str],
) -> None:
    """
    Replace placeholder tokens inside all text files under the root directory.

    Parameters
    ----------
    root : Path
        Root path of the scaffolded project directory.
    placeholders : dict[str, str]
        Dictionary mapping placeholder key strings to replacement values.
    """
    # Iterate through all regular files under root directory
    for file in find_files(root):
        # Skip binary files to prevent encoding corruption
        if not is_text_file(file):
            continue

        content = read_text(file)

        # Substitute each placeholder key with its concrete value
        for placeholder, value in placeholders.items():
            content = content.replace(placeholder, value)

        write_text(file, content)


def replace_placeholders_in_paths(
    root: Path,
    placeholders: dict[str, str],
) -> None:
    """
    Rename directory and file paths containing placeholder tokens.

    Parameters
    ----------
    root : Path
        Root path of the scaffolded project directory.
    placeholders : dict[str, str]
        Dictionary mapping placeholder key strings to replacement values.

    Notes
    -----
    Deepest paths (longest path parts) are renamed first to avoid invalidating parent
    directory path references during traversal.
    """
    # Sort paths in descending order of path depth
    paths = sorted(
        find_directories(root) + find_files(root),
        key=lambda p: len(p.parts),
        reverse=True,
    )

    for path in paths:
        new_name = path.name

        for placeholder, value in placeholders.items():
            new_name = new_name.replace(placeholder, value)

        # Rename path if filename/directory name was modified
        if new_name != path.name:
            rename_path(path, path.with_name(new_name))


def replace_all(
    root: Path,
    placeholders: dict[str, str],
) -> None:
    """
    Execute placeholder replacement on both file paths and file contents.

    Parameters
    ----------
    root : Path
        Root path of the scaffolded project directory.
    placeholders : dict[str, str]
        Dictionary mapping placeholder key strings to replacement values.
    """
    # First rename paths containing placeholder tokens
    replace_placeholders_in_paths(root, placeholders)
    # Next substitute placeholder strings inside file text content
    replace_placeholders_in_content(root, placeholders)
