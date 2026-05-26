# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pathlib

from py4vasp import raw
from py4vasp._calculation.dispatch import (
    DataSource,
    FileSource,
    merge_default,
    merge_strings,
    quantity,
)
from py4vasp._util import convert


class SystemHandler:
    """Handler for the system tag. Works with exactly one raw.System object."""

    def __init__(self, raw_system: raw.System):
        self._raw_system = raw_system

    @classmethod
    def from_data(cls, raw_system: raw.System) -> "SystemHandler":
        return cls(raw_system)

    def read(self) -> dict:
        """Read the system tag into a dictionary."""
        return {"system": str(self)}

    def to_dict(self) -> dict:
        """Public alias for read()."""
        return self.read()

    def __str__(self) -> str:
        return convert.text_to_string(self._raw_system.system)


@quantity("system")
class System:
    """The :tag:`SYSTEM` tag in the INCAR file is a title you choose for a VASP calculation.

    VASP lets you attach a free-form description to every calculation via the
    :tag:`SYSTEM` tag in the INCAR file. This class provides access to that
    string. It has no physical significance, but is useful for bookkeeping
    when managing many calculations.

    Examples
    --------
    Print the system tag of a calculation:

    >>> from py4vasp import demo
    >>> calculation = demo.calculation(path)
    >>> print(calculation.system)
    Sr2TiO4 calculation
    """

    def __init__(self, source, quantity_name: str = "system"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_system: raw.System) -> "System":
        """Create a System dispatcher from raw data (convenience for testing)."""
        return cls(source=DataSource(raw_system))

    @classmethod
    def from_path(cls, path=".") -> "System":
        """Create a System dispatcher that reads from HDF5 files at *path*."""
        return cls(source=FileSource(path))

    @classmethod
    def from_file(cls, file_name) -> "System":
        """Create a System dispatcher that reads from a specific HDF5 file."""
        resolved = pathlib.Path(file_name).expanduser().resolve()
        return cls(source=FileSource(resolved.parent, file=file_name))

    def read(self, selection: str | None = None) -> dict:
        """Read the system tag into a dictionary.

        Returns
        -------
        -
            A dictionary with a single key ``"system"`` whose value is the
            title string set by the :tag:`SYSTEM` tag in the INCAR file.

        Examples
        --------
        Read the system tag of a calculation:

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> calculation.system.read()
        {'system': 'Sr2TiO4 calculation'}
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            SystemHandler.from_data,
            SystemHandler.read,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Read the system tag into a dictionary.

        Convenient alias for :py:meth:`read`. Check that method for examples
        and optional arguments.

        Examples
        --------
        Read the system tag of a calculation:

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> calculation.system.to_dict()
        {'system': 'Sr2TiO4 calculation'}
        """
        return self.read(selection=selection)

    def __str__(self) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            None,
            SystemHandler.from_data,
            SystemHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
