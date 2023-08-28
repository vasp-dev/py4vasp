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
