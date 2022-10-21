# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import data
from py4vasp.data import _base


class Dispersion(_base.Refinery):
    """Generic class for all dispersions (electrons, phonons).

    Provides some utility functionalities common to all dispersions to avoid duplication
    of code."""

    def to_dict(self):
        return {
            "kpoint_distances": self._kpoints.distances(),
            "kpoint_labels": self._kpoints.labels(),
            "eigenvalues": self._raw_data.eigenvalues[:],
        }

    @property
    def _kpoints(self):
        return data.Kpoint.from_data(self._raw_data.kpoints)
