# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import data
from py4vasp.data import _base
import py4vasp._third_party.graph as _graph


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

    def plot(self):
        return _graph.Graph(series=list(self._band_structure()))

    def _band_structure(self):
        data = self.to_dict()
        bands = _make_3d(data["eigenvalues"])
        for component in bands:
            yield _graph.Series(data["kpoint_distances"], component.T)


def _make_3d(eigenvalues):
    if eigenvalues.ndim == 2:
        return eigenvalues[None, :, :]
    else:
        return eigenvalues
