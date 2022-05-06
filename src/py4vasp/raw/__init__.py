# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
""" Extract the raw data from the HDF5 file and transform it into dataclasses.

In the HDF5 file, the raw data is stored with specific keys. To avoid
propagating the name of these keys to the higher tier modules, we transform
everything into dataclasses. This enables the introduction of new file formats by
replacing the `File` class.

Notes
-----
The data from the HDF5 file is lazily loaded except for scalars. This avoids
memory issues when the HDF5 file contains a lot of data, because only what is
needed is read. However, this has the consequence that you need to
enforce the read operation before the file is closed.
"""

from .data_wrapper import VaspData
from .rawdata import *
from .data import *
from ._access import access
from .file import File
import inspect
import sys

_this_mod = sys.modules[__name__]
__all__ = [name for name, _ in inspect.getmembers(_this_mod, inspect.isclass)]
