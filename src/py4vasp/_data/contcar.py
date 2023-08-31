# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import data, raw
from py4vasp._data import base, structure
from py4vasp._util import convert


class CONTCAR(base.Refinery, structure.Mixin):
    "Access the final positions after the VASP calculation."

    def to_dict(self):
        return {
            **self._structure.read(),
            "system": convert.text_to_string(self._raw_data.system),
            **self._read("selective_dynamics"),
            **self._read("lattice_velocities"),
            **self._read("ion_velocities"),
        }

    def _read(self, key):
        data = getattr(self._raw_data, key)
        return {key: data} if not data.is_none() else {}
