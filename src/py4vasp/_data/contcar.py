# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import data, raw
from py4vasp._data import base
from py4vasp._util import convert


class CONTCAR(base.Refinery):
    "Access the final positions after the VASP calculation."

    def to_dict(self):
        return {
            **self._structure().read(),
            "system": convert.text_to_string(self._raw_data.system),
            "selective_dynamics": self._read_selective_dynamics(),
            "lattice_velocities": self._read_lattice_velocities(),
            "ion_velocities": self._read_ion_velocities(),
        }

    def _structure(self):
        structure = self._raw_data.structure
        raw_structure = raw.Structure(
            cell=raw.Cell(
                lattice_vectors=np.array([structure.cell.lattice_vectors]),
                scale=structure.cell.scale,
            ),
            topology=structure.topology,
            positions=np.array([structure.positions]),
        )
        return data.Structure.from_data(raw_structure)

    def _read_selective_dynamics(self):
        if self._raw_data.selective_dynamics.is_none():
            return None
        else:
            return self._raw_data.selective_dynamics[:]

    def _read_lattice_velocities(self):
        if self._raw_data.lattice_velocities.is_none():
            return None
        else:
            return self._raw_data.lattice_velocities[:]

    def _read_ion_velocities(self):
        if self._raw_data.ion_velocities.is_none():
            return None
        else:
            return self._raw_data.ion_velocities[:]
