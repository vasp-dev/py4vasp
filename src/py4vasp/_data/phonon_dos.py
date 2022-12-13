# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import data
from py4vasp._data import base
from py4vasp._data.phonon_projector import PhononProjector, selection_doc
from py4vasp._third_party import graph
from py4vasp._util import documentation


class PhononDos(base.Refinery, graph.Mixin):
    """The phonon density of states (DOS).

    You can use this class to extract the phonon DOS data of a VASP
    calculation. The DOS can also be resolved by direction and atom.
    """

    @base.data_access
    def __str__(self):
        energies = self._raw_data.energies
        projector = self._get_projector()
        return f"""phonon DOS:
    [{energies[0]:0.2f}, {energies[-1]:0.2f}] mesh with {len(energies)} points
    {projector.modes} modes
    {self._topology}"""

    @base.data_access
    @documentation.format(selection=selection_doc)
    def to_dict(self, selection=None):
        """Read the phonon DOS into a dictionary.

        Parameters
        ----------
        {selection}

        Returns
        -------
        dict
            Contains the energies at which the phonon  DOS was computed. The total
            DOS is returned and any possible projected DOS selected by the *selection*
            argument.
        """
        return {
            "energies": self._raw_data.energies[:],
            "total": self._raw_data.dos[:],
            **self._read_data(selection),
        }

    @base.data_access
    @documentation.format(selection=selection_doc)
    def to_graph(self, selection=None):
        """Generate a graph of the selected DOS.

        Parameters
        ----------
        {selection}

        Returns
        -------
        Graph
            The graph contains the total DOS. If a selection is given, in addition the
            projected DOS is shown."""
        data = self.to_dict(selection)
        return graph.Graph(
            series=list(_series(data)),
            xlabel="ω (THz)",
            ylabel="DOS (1/THz)",
        )

    @property
    def _topology(self):
        return data.Topology.from_data(self._raw_data.topology)

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


def _series(data):
    energies = data["energies"]
    for name, dos in data.items():
        if name == "energies":
            continue
        yield graph.Series(energies, dos, name)
