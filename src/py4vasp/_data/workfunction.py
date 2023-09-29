# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._data import base


class Workfunction(base.Refinery):
    """The workfunction of a material describes the energy required to remove an electron
    to the vacuum.

    In VASP you can compute the workfunction by setting the IDIPOL flag in the INCAR file.
    This class provides then the functionality to analyze the resulting potential."""
