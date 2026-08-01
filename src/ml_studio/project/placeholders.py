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
    Replace placeholders inside every text file.
    """

    for file in find_files(root):
        if not is_text_file(file):
            continue

        content = read_text(file)

        for placeholder, value in placeholders.items():
            content = content.replace(placeholder, value)

        write_text(file, content)


def replace_placeholders_in_paths(
    root: Path,
    placeholders: dict[str, str],
) -> None:
    """
    Rename directories and files containing placeholders.

    Deepest paths are renamed first to avoid invalid parent paths.
    """

    paths = sorted(
        find_directories(root) + find_files(root),
        key=lambda p: len(p.parts),
        reverse=True,
    )

    for path in paths:
        new_name = path.name

        for placeholder, value in placeholders.items():
            new_name = new_name.replace(placeholder, value)

        if new_name != path.name:
            rename_path(path, path.with_name(new_name))


def replace_all(
    root: Path,
    placeholders: dict[str, str],
) -> None:
    """
    Replace placeholders in both paths and file contents.
    """

    replace_placeholders_in_paths(root, placeholders)
    replace_placeholders_in_content(root, placeholders)
