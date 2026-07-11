from pathlib import Path

from scripts.project.file_utils import (
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
    source = tmp_path / "src"
    destination = tmp_path / "dst"

    source.mkdir()
    (source / "test.txt").write_text("hello")

    copy_directory(source, destination)

    assert (destination / "test.txt").exists()


def test_rename_path(tmp_path: Path) -> None:
    src = tmp_path / "old"
    dst = tmp_path / "new"

    src.mkdir()

    rename_path(src, dst)

    assert dst.exists()


def test_find_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")

    files = find_files(tmp_path)

    assert len(files) == 2


def test_find_directories(tmp_path: Path) -> None:
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()

    directories = find_directories(tmp_path)

    assert len(directories) == 2


def test_read_write(tmp_path: Path) -> None:
    file = tmp_path / "hello.txt"

    write_text(file, "hello")

    assert read_text(file) == "hello"


def test_is_text_file(tmp_path: Path) -> None:
    file = tmp_path / "test.txt"

    file.write_text("hello")

    assert is_text_file(file)


def test_delete_directory(tmp_path: Path) -> None:
    directory = tmp_path / "temp"

    directory.mkdir()

    delete_directory(directory)

    assert not directory.exists()