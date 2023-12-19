# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import data
from py4vasp._data import base, phonon
from py4vasp._third_party import graph
from py4vasp._util import convert, documentation, index, select


class PhononBand(phonon.Mixin, base.Refinery, graph.Mixin):
    """The phonon band structure.

    Use this to examine the phonon band structure along a high-symmetry path in the
    Brillouin zone. The `to_dict` function allows to extract the raw data to process
    it further."""

    @base.data_access
    def __str__(self):
        return f"""phonon band data:
    {self._raw_data.dispersion.eigenvalues.shape[0]} q-points
    {self._raw_data.dispersion.eigenvalues.shape[1]} modes
    {self._topology()}"""

    @base.data_access
    def to_dict(self):
        """Read the phonon band structure into a dictionary.

        Returns
        -------
        dict
            Contains the **q**-point path for plotting phonon band structures and
            the phonon bands. In addition the phonon modes are returned.
        """
        dispersion = self._dispersion().read()
        return {
            "qpoint_distances": dispersion["kpoint_distances"],
            "qpoint_labels": dispersion["kpoint_labels"],
            "bands": dispersion["eigenvalues"],
            "modes": self._modes(),
        }

    @base.data_access
    @documentation.format(selection=phonon.selection_doc)
    def to_graph(self, selection=None, width=1.0):
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
        graph = self._dispersion().plot(projections)
        graph.ylabel = "ω (THz)"
        return graph

    def _dispersion(self):
        return data.Dispersion.from_data(self._raw_data.dispersion)

    def _modes(self):
        return convert.to_complex(self._raw_data.eigenvectors[:])

    def _projections(self, selection, width):
        if not selection:
            return None
        maps = {2: self._init_atom_dict(), 3: self._init_direction_dict()}
        selector = index.Selector(maps, np.abs(self._modes()), use_number_labels=True)
        tree = select.Tree.from_selection(selection)
        return {
            selector.label(selection): width * selector[selection]
            for selection in tree.selections()
        }
