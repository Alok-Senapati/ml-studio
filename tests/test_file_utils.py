"""
Unit tests for file utility and directory manipulation functions.
"""

from __future__ import annotations

from pathlib import Path

from ml_studio.project.file_utils import (
    copy_directory,
    delete_directory,
    find_directories,
    find_files,
    is_text_file,
    read_text,
    rename_path,
    write_text,
)


def test_copy_directory(tmp_path: Path) -> None:
    """Test recursive directory copying."""
    source = tmp_path / "src"
    destination = tmp_path / "dst"

    source.mkdir()
    (source / "test.txt").write_text("hello", encoding="utf-8")

    copy_directory(source, destination)

    assert (destination / "test.txt").exists()


def test_rename_path(tmp_path: Path) -> None:
    """Test path renaming functionality."""
    src = tmp_path / "old"
    dst = tmp_path / "new"

    src.mkdir()
    rename_path(src, dst)

    assert dst.exists()


def test_find_files(tmp_path: Path) -> None:
    """Test file search traversal."""
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")

    files = find_files(tmp_path)

    assert len(files) == 2


def test_find_directories(tmp_path: Path) -> None:
    """Test directory search traversal."""
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()

    directories = find_directories(tmp_path)

    assert len(directories) == 2


def test_read_write(tmp_path: Path) -> None:
    """Test text reading and writing helpers."""
    file = tmp_path / "hello.txt"

    write_text(file, "hello")

    assert read_text(file) == "hello"


def test_is_text_file(tmp_path: Path) -> None:
    """Test text file encoding validation."""
    file = tmp_path / "test.txt"
    file.write_text("hello", encoding="utf-8")

    assert is_text_file(file)


def test_delete_directory(tmp_path: Path) -> None:
    """Test recursive directory deletion."""
    directory = tmp_path / "temp"
    directory.mkdir()

    delete_directory(directory)

    assert not directory.exists()


def test_is_text_file_binary(tmp_path: Path) -> None:
    """Test that binary byte sequences trigger False in is_text_file check."""
    file = tmp_path / "binary.bin"
    file.write_bytes(b"\xff\xff\xff\xff")

    assert not is_text_file(file)
