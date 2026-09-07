# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import io
import pathlib
import tarfile
import zipfile

import pytest

from py4vasp import exception
from py4vasp._util import archive

EXAMPLE_FILES = {
    "run/INCAR": "ISMEAR = 0",
    "run/vaspout.h5": "not really HDF5",
    "run/subdirectory/POSCAR": "not really a POSCAR",
}
FORMATS = ("zip", "tar", "tar.gz", "tar.bz2", "tar.xz")


def make_archive(directory, format_, files=None, name="example"):
    """Create an archive of the given format and return the path to it."""
    files = EXAMPLE_FILES if files is None else files
    filename = pathlib.Path(directory) / f"{name}.{format_}"
    if format_ == "zip":
        with zipfile.ZipFile(filename, "w") as zip_file:
            for member, content in files.items():
                zip_file.writestr(member, content)
    else:
        compression = format_.partition(".")[2]
        with tarfile.open(filename, f"w:{compression}") as tar_file:
            for member, content in files.items():
                _add_to_tar(tar_file, member, content)
    return filename


def _add_to_tar(tar_file, member, content):
    data = content.encode()
    info = tarfile.TarInfo(member)
    info.size = len(data)
    tar_file.addfile(info, io.BytesIO(data))


@pytest.fixture(params=FORMATS)
def example_archive(request, tmp_path):
    return make_archive(tmp_path, request.param)


@pytest.fixture
def not_an_archive(tmp_path):
    filename = tmp_path / "not_an_archive.zip"
    filename.write_text("This is a text file and not an archive.")
    return filename


def test_members_of_archive(example_archive):
    with archive.open_archive(example_archive) as opened:
        actual = {str(member) for member in opened.members()}
    assert actual == set(EXAMPLE_FILES)


def test_directories_are_not_members(tmp_path):
    filename = tmp_path / "with_directories.zip"
    with zipfile.ZipFile(filename, "w") as zip_file:
        zip_file.writestr("run/", "")
        zip_file.writestr("run/vaspout.h5", "content")
    with archive.open_archive(filename) as opened:
        assert [str(member) for member in opened.members()] == ["run/vaspout.h5"]


def test_format_is_determined_from_content(tmp_path):
    # a zip archive with the name of a tar archive is still readable
    misnamed = make_archive(tmp_path, "zip", name="misnamed")
    misnamed = misnamed.rename(tmp_path / "misnamed.tgz")
    with archive.open_archive(misnamed) as opened:
        assert len(opened.members()) == len(EXAMPLE_FILES)


def test_file_that_is_not_an_archive(not_an_archive):
    with pytest.raises(exception.FileAccessError) as error:
        with archive.open_archive(not_an_archive):
            pass
    assert "not an archive" in str(error.value)


def test_missing_archive(tmp_path):
    with pytest.raises(exception.FileAccessError) as error:
        with archive.open_archive(tmp_path / "does_not_exist.zip"):
            pass
    assert "exists" in str(error.value)


def test_directory_instead_of_archive(tmp_path):
    with pytest.raises(exception.FileAccessError):
        with archive.open_archive(tmp_path):
            pass


@pytest.mark.parametrize(
    "junk", ("__MACOSX/run/._vaspout.h5", "run/.DS_Store", "run/._INCAR", "Thumbs.db")
)
def test_junk_files_are_ignored(tmp_path, junk):
    files = {**EXAMPLE_FILES, junk: "junk"}
    filename = make_archive(tmp_path, "zip", files=files)
    with archive.open_archive(filename) as opened:
        actual = {str(member) for member in opened.members()}
    assert actual == set(EXAMPLE_FILES)


@pytest.mark.parametrize("unsafe", ("../evil.txt", "run/../../evil.txt", "/evil.txt"))
def test_unsafe_members_are_ignored(tmp_path, unsafe):
    files = {**EXAMPLE_FILES, unsafe: "evil"}
    filename = make_archive(tmp_path, "tar", files=files)
    with archive.open_archive(filename) as opened:
        actual = {str(member) for member in opened.members()}
    assert actual == set(EXAMPLE_FILES)


def test_links_are_ignored(tmp_path):
    filename = tmp_path / "with_link.tar"
    with tarfile.open(filename, "w") as tar_file:
        _add_to_tar(tar_file, "run/vaspout.h5", "content")
        info = tarfile.TarInfo("run/link.h5")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../../etc/passwd"
        tar_file.addfile(info)
    with archive.open_archive(filename) as opened:
        assert [str(member) for member in opened.members()] == ["run/vaspout.h5"]


def test_is_archive(example_archive):
    assert archive.is_archive(example_archive)


def test_is_not_an_archive(not_an_archive, tmp_path):
    assert not archive.is_archive(not_an_archive)
    assert not archive.is_archive(tmp_path / "does_not_exist.zip")
    assert not archive.is_archive(tmp_path)


def test_extract_files(example_archive, tmp_path):
    destination = tmp_path / "destination"
    destination.mkdir()
    with archive.open_archive(example_archive) as opened:
        members = [member for member in opened.members() if member.name != "INCAR"]
        opened.extract(members, destination)
    # the directory structure of the archive is not reproduced
    assert sorted(path.name for path in destination.iterdir()) == [
        "POSCAR",
        "vaspout.h5",
    ]
    assert (destination / "vaspout.h5").read_text() == EXAMPLE_FILES["run/vaspout.h5"]
    assert (destination / "POSCAR").read_text() == EXAMPLE_FILES[
        "run/subdirectory/POSCAR"
    ]


def test_extract_nothing(example_archive, tmp_path):
    destination = tmp_path / "destination"
    destination.mkdir()
    with archive.open_archive(example_archive) as opened:
        opened.extract([], destination)
    assert list(destination.iterdir()) == []


MARKERS = ("vaspout.h5", "vaspwave.h5")
FLAT_FILES = {"INCAR": "ISMEAR = 0", "vaspout.h5": "not really HDF5"}
NESTED_FILES = {"outer/run/INCAR": "ISMEAR = 0", "outer/run/vaspout.h5": "no HDF5"}
MULTIPLE_FILES = {
    "relax/vaspout.h5": "no HDF5",
    "static/vaspout.h5": "no HDF5",
    "bands/vaspwave.h5": "no HDF5",
    "documentation/README": "not a calculation",
}


def select_directory(filename, path=None):
    with archive.open_archive(filename) as opened:
        return archive.select_directory(opened, MARKERS, path=path)


@pytest.mark.parametrize(
    "files, expected",
    (
        (FLAT_FILES, "."),
        (EXAMPLE_FILES, "run"),
        (NESTED_FILES, "outer/run"),
    ),
)
def test_select_only_directory(tmp_path, files, expected):
    filename = make_archive(tmp_path, "zip", files=files)
    assert select_directory(filename) == pathlib.PurePosixPath(expected)


@pytest.mark.parametrize("path", ("relax", "./relax", "relax/", pathlib.Path("relax")))
def test_select_directory_by_path(tmp_path, path):
    filename = make_archive(tmp_path, "zip", files=MULTIPLE_FILES)
    assert select_directory(filename, path) == pathlib.PurePosixPath("relax")


def test_select_root_by_path(tmp_path):
    filename = make_archive(tmp_path, "zip", files=FLAT_FILES)
    assert select_directory(filename, ".") == pathlib.PurePosixPath(".")


def test_select_directory_without_marker(tmp_path):
    # a directory may be selected even if it does not contain VASP output
    filename = make_archive(tmp_path, "zip", files=MULTIPLE_FILES)
    expected = pathlib.PurePosixPath("documentation")
    assert select_directory(filename, "documentation") == expected


def test_multiple_directories_raise_error(tmp_path):
    filename = make_archive(tmp_path, "zip", files=MULTIPLE_FILES)
    with pytest.raises(exception.IncorrectUsage) as error:
        select_directory(filename)
    message = str(error.value)
    assert all(directory in message for directory in ("relax", "static", "bands"))
    assert "documentation" not in message


def test_archive_without_calculation_raises_error(tmp_path):
    files = {"documentation/README": "not a calculation"}
    filename = make_archive(tmp_path, "zip", files=files)
    with pytest.raises(exception.FileAccessError) as error:
        select_directory(filename)
    assert "documentation" in str(error.value)


def test_path_not_in_archive_raises_error(tmp_path):
    filename = make_archive(tmp_path, "zip", files=MULTIPLE_FILES)
    with pytest.raises(exception.IncorrectUsage) as error:
        select_directory(filename, "does_not_exist")
    message = str(error.value)
    assert "does_not_exist" in message
    assert "relax" in message
