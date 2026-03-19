# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation import base
from py4vasp._raw import data as raw_data
from py4vasp._util import convert


class System(base.Refinery):
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

    _raw_data: raw_data.System

    @base.data_access
    def __str__(self) -> str:
        return convert.text_to_string(self._raw_data.system)

    @base.data_access
    def to_dict(self) -> dict[str, str]:
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
        >>> calculation.system.to_dict()
        {'system': 'Sr2TiO4 calculation'}
        """
        return {"system": str(self)}
