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
            "modes": self._modes,
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
        projections = self._projections(selection, width)
        graph = self._dispersion.plot(projections)
        graph.ylabel = "ω (THz)"
        return graph

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
    def _topology(self):
        return data.Topology.from_data(self._raw_data.topology)

    @property
    def _modes(self):
        return _convert.to_complex(self._raw_data.eigenvectors[:])

    @property
    def _projector(self):
        topology = data.Topology.from_data(self._raw_data.topology)
        return PhononProjector(topology)

    def _projections(self, selection, width):
        if selection is None:
            return None
        projector = self._projector
        return dict(
            self._create_projection(*projector.select(*index), width)
            for index in projector.parse_selection(selection)
        )

    def _create_projection(self, label, select, width):
        selected = self._modes[:, :, select.atom.indices, select.direction.indices]
        return label, width * np.sum(np.abs(selected), axis=(2, 3))
