# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
from py4vasp import data
from py4vasp.data import _base, _export
from py4vasp.data._phonon_projector import PhononProjector, selection_doc
from py4vasp._util import convert as _convert, documentation
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
        """Read the phonon band structure into a dictionary.

        Returns
        -------
        dict
            Contains the **q**-point path for plotting phonon band structures and
            the phonon bands. In addition the phonon modes are returned.
        """
        dispersion = self._dispersion.read()
        return {
            "qpoint_distances": dispersion["kpoint_distances"],
            "qpoint_labels": dispersion["kpoint_labels"],
            "bands": dispersion["eigenvalues"],
            "modes": _convert.to_complex(self._raw_data.eigenvectors[:]),
        }

    @_base.data_access
    @documentation.format(selection=selection_doc)
    def plot(self, selection=None, width=1.0):
        """Generate a graph of the phonon bands.

        Parameters
        ----------
        {selection}
        width : float
            Specifies the width illustrating the projections.

        Returns
        -------
        Graph
            Contains the phonon band structure for all the **q** points. If a
            selection is provided, the width of the bands is adjusted according to
            the projection.
        """
        return _graph.Graph(
            series=self._band_structure(selection, width),
            xticks=self._xticks(),
            ylabel="ω (THz)",
        )

    @_base.data_access
    def to_plotly(self, selection=None, width=1.0):
        """Generate a plotly figure of the phonon band structure.

        Converts the Graph object to a plotly figure. Check the :py:meth:`plot` method
        to learn about how the Graph is generated."""
        return self.plot(selection, width).to_plotly()

    @property
    def _dispersion(self):
        return data.Dispersion.from_data(self._raw_data.dispersion)

    @property
    def _qpoints(self):
        return data.Kpoint.from_data(self._raw_data.dispersion.kpoints)

    @property
    def _topology(self):
        return data.Topology.from_data(self._raw_data.topology)

    def _xticks(self):
        ticks, labels = self._degenerate_ticks_and_labels()
        return self._filter_unique(ticks, labels)

    def _degenerate_ticks_and_labels(self):
        labels = self._qpoint_labels()
        mask = np.logical_or(self._edge_of_line(), labels != "")
        return self._qpoints.distances()[mask], labels[mask]

    def _qpoint_labels(self):
        labels = self._qpoints.labels()
        if labels is None:
            labels = [""] * len(self._raw_data.dispersion.kpoints.coordinates)
        return np.array(labels)

    def _edge_of_line(self):
        indices = np.arange(len(self._raw_data.dispersion.kpoints.coordinates))
        edge_of_line = (indices + 1) % self._qpoints.line_length() == 0
        edge_of_line[0] = True
        return edge_of_line

    def _filter_unique(self, ticks, labels):
        result = {}
        for tick, label in zip(ticks, labels):
            if tick in result:
                previous_label = result[tick]
                if previous_label != "" and previous_label != label:
                    label = previous_label + "|" + label
            result[tick] = label
        return result

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
