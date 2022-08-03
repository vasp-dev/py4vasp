# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
from py4vasp import data
from py4vasp.data import _base, _export
from py4vasp.data._phonon_projector import PhononProjector
from py4vasp._util import convert as _convert
import py4vasp._third_party.graph as _graph


class PhononBand(_base.Refinery, _export.Image):
    """The phonon band structure.

    Use this to examine the phonon band structure along a high-symmetry path in the
    Brillouin zone. The `to_dict` function allows to extract the raw data to process
    it further."""

    @_base.data_access
    def __str__(self):
        return f"""phonon band data:
    {self._raw_data.dispersion.eigenvalues.shape[0]} q-points
    {self._raw_data.dispersion.eigenvalues.shape[1]} modes
    {self._topology}"""

    @_base.data_access
    def to_dict(self):
        return {
            "qpoint_distances": self._qpoints.distances(),
            "bands": self._raw_data.dispersion.eigenvalues[:],
            "modes": _convert.to_complex(self._raw_data.eigenvectors[:]),
        }

    @_base.data_access
    def plot(self, selection=None, width=1.0):
        return _graph.Graph(
            series=self._band_structure(selection, width),
            ylabel="ω (THz)",
        )

    @_base.data_access
    def to_plotly(self, selection=None, width=1.0):
        return self.plot(selection, width).to_plotly()

    @property
    def _qpoints(self):
        return data.Kpoint.from_data(self._raw_data.dispersion.kpoints)

    @property
    def _topology(self):
        return data.Topology.from_data(self._raw_data.topology)

    def _band_structure(self, selection, width):
        band = self.to_dict()
        if selection is None:
            return self._regular_band_structure(band)
        else:
            return self._fat_band_structure(band, selection, width)

    def _regular_band_structure(self, band):
        return [_graph.Series(x=band["qpoint_distances"], y=band["bands"].T)]

    def _fat_band_structure(self, band, selection, width):
        projector = self._get_projector()
        result = []
        for index in projector.parse_selection(selection):
            label, selection = projector.select(*index)
            result.append(self._fat_band(band, label, selection, width))
        return result

    def _get_projector(self):
        topology = data.Topology.from_data(self._raw_data.topology)
        return PhononProjector(topology)

    def _fat_band(self, band, label, selection, width):
        selected = band["modes"][
            :, :, selection.atom.indices, selection.direction.indices
        ]
        return _graph.Series(
            x=band["qpoint_distances"],
            y=band["bands"].T,
            name=label,
            width=width * np.sum(np.abs(selected), axis=(2, 3)).T,
        )
