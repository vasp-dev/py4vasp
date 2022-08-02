# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
from py4vasp import data
from py4vasp.data import _base
from py4vasp.data._phonon_projector import PhononProjector


class PhononDos(_base.Refinery):
    """The phonon density of states (DOS).

    You can use this class to extract the phonon DOS data of a VASP
    calculation. The DOS can also be resolved by direction and atom.
    """

    def to_dict(self, selection=None):
        return {
            "energies": self._raw_data.energies[:],
            "total": self._raw_data.dos[:],
            **self._read_data(selection),
        }

    def _read_data(self, selection):
        projector = self._get_projector()
        result = {}
        for index in projector.parse_selection(selection):
            label, selection = projector.select(*index)
            result[label] = self._partial_dos(selection)
        return result

    def _get_projector(self):
        topology = data.Topology.from_data(self._raw_data.topology)
        return PhononProjector(topology)

    def _partial_dos(self, selection):
        projections = self._raw_data.projections[
            selection.atom.indices, selection.direction.indices
        ]
        return np.sum(projections, axis=(0, 1))
