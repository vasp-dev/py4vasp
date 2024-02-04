# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np

from py4vasp import data, raw
from py4vasp._data import base, structure
from py4vasp._util import convert

INDEXING_OSZICAR = {
    "iteration_number": 0,
    "free_energy": 1,
    "free_energy_change": 2,
    "bandstructure_energy_change": 3,
    "number_hamiltonian_evaluations": 4,
    "norm_residual": 5,
    "difference_charge_density": 6,
}


class OSZICAR(base.Refinery, structure.Mixin):
    """Access the convergence data for each electronic step."""

    @base.data_access
    def to_dict(self):
        return_data = {}
        for key in INDEXING_OSZICAR:
            return_data[key] = self._read(key)
        return return_data

    def _read(self, key):
        data = getattr(self._raw_data, "convergence_data")
        data_index = INDEXING_OSZICAR[key]
        return data[:, data_index] if not data.is_none() else {}
