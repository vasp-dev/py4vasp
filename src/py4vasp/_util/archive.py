# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Read the content of an archive in which a VASP calculation is stored.

Users frequently archive a finished calculation as a zip or tar file before moving it
off a cluster. This module provides the minimal machinery to look inside such an archive
and to copy individual files out of it. It deliberately knows nothing about VASP; the
caller decides which filenames are interesting.

Only the standard library is used so that this module also works with the py4vasp-core
package, which depends on numpy and h5py alone.
"""

import contextlib
import pathlib
import tarfile
import zipfile

from py4vasp import exception

# Some tools add bookkeeping files to an archive. They are never part of a VASP
# calculation, but they would make an archive look like it contained more directories
# than it does, so we filter them out everywhere.
_JUNK_DIRECTORY = "__MACOSX"
_JUNK_FILENAMES = (".DS_Store", "Thumbs.db")
_JUNK_PREFIX = "._"

_FORMATS = "zip, tar, tar.gz (tgz), tar.bz2, and tar.xz"


def is_archive(filename):
    """Check whether the given file is an archive that py4vasp can read.

    The format is determined from the content of the file and not from its suffix, so
    that an archive with an unusual name is recognized, too.

    Parameters
    ----------
    filename : str or pathlib.Path
        Name of the file that may be an archive.

    Returns
    -------
    bool
        True if the file is a zip or tar archive.
    """
    try:
        return zipfile.is_zipfile(filename) or tarfile.is_tarfile(filename)
    except (OSError, ValueError):
        return False


@contextlib.contextmanager
def open_archive(filename):
    """Open an archive to inspect its content and extract files from it.

    Parameters
    ----------
    filename : str or pathlib.Path
        Name of the zip or tar archive.

    Returns
    -------
    ContextManager
        Entering the context manager yields an archive from which the files can be read.
        The archive is closed when the context terminates.
    """
    filename = pathlib.Path(filename).expanduser().resolve()
    if not filename.is_file():
        message = f"{filename} could not be opened. Please make sure the file exists."
        raise exception.FileAccessError(message) from None
    with contextlib.ExitStack() as stack:
        yield _create_archive(filename, stack)


def _create_archive(filename, stack):
    # An error of the caller must not be converted, so only the setup of the archive is
    # wrapped here and not the body of the context manager above.
    try:
        if zipfile.is_zipfile(filename):
            return _ZipArchive(filename, stack.enter_context(zipfile.ZipFile(filename)))
        if tarfile.is_tarfile(filename):
            return _TarArchive(filename, stack.enter_context(tarfile.open(filename)))
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as error:
        message = (
            f"Error when reading the archive {filename}. Please check whether the file "
            "is a valid archive and that you have the permissions to read it."
        )
        raise exception.FileAccessError(message) from error
    message = f"""\
{filename} is not an archive that py4vasp can read. The supported formats are
{_FORMATS}. py4vasp determines the format from the content of the file, so renaming
the file does not change which archives it can read."""
    raise exception.FileAccessError(message)


class _Archive:
    """Uniform read-only view on the files stored in an archive.

    The subclasses provide the format specific part: how the entries of the archive are
    obtained and how a single one of them is opened for reading.
    """

    def __init__(self, filename):
        self.filename = filename
        self._members = {}
        for name, entry in self._raw_entries():
            member = _safe_member(name)
            if member is not None:
                self._members[member] = entry

    def members(self):
        """Return the files in the archive as a tuple of pathlib.PurePosixPath."""
        return tuple(self._members)

    def _raw_entries(self):
        raise NotImplementedError

    def _open_member(self, entry):
        raise NotImplementedError


class _ZipArchive(_Archive):
    def __init__(self, filename, zip_file):
        self._zip_file = zip_file
        super().__init__(filename)

    def _raw_entries(self):
        for info in self._zip_file.infolist():
            if not info.is_dir():
                yield info.filename, info

    def _open_member(self, entry):
        return self._zip_file.open(entry)


class _TarArchive(_Archive):
    def __init__(self, filename, tar_file):
        self._tar_file = tar_file
        super().__init__(filename)

    def _raw_entries(self):
        for member in self._tar_file.getmembers():
            # isfile excludes directories as well as symlinks, hardlinks, and device
            # nodes. Those could point outside of the extraction directory, so they are
            # never considered part of the calculation.
            if member.isfile():
                yield member.name, member

    def _open_member(self, entry):
        return self._tar_file.extractfile(entry)


def _safe_member(name):
    """Convert an entry of the archive to a path rejecting anything suspicious.

    An archive may contain a path that leaves the directory into which it is extracted.
    py4vasp ignores these files instead of overwriting data elsewhere on the disk. An
    archive is also not required to use the path separator of the current platform, so
    a backslash in the filename is rejected as well.
    """
    member = pathlib.PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts:
        return None
    if _JUNK_DIRECTORY in member.parts:
        return None
    filename = member.name
    if not filename or "\\" in filename:
        return None
    if filename in _JUNK_FILENAMES or filename.startswith(_JUNK_PREFIX):
        return None
    return member
