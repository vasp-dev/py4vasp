# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import data
from py4vasp.data import _base
import py4vasp._third_party.graph as _graph


class PhononBand(_base.Refinery):
    """The phonon band structure.

    Use this to examine the phonon band structure along a high-symmetry path in the
    Brillouin zone. The `to_dict` function allows to extract the raw data to process
    it further."""

    @_base.data_access
    def to_dict(self):
        return {
            "bands": self._raw_data.dispersion.eigenvalues[:],
            "modes": self._raw_data.eigenvectors[:],
        }

    @_base.data_access
    def plot(self):
        band_structure = _graph.Series(
            x=self._qpoints.distances(), y=self._raw_data.dispersion.eigenvalues[:].T
        )
        return _graph.Graph(
            series=[band_structure],
            ylabel="Energy (meV)",
        )

    @property
    def _qpoints(self):
        return data.Kpoint.from_data(self._raw_data.dispersion.kpoints)
