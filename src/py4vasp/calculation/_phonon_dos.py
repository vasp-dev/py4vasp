# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._third_party import graph
from py4vasp._util import documentation, index, select
from py4vasp.calculation import _base, _phonon


class PhononDos(_phonon.Mixin, _base.Refinery, graph.Mixin):
    """The phonon density of states (DOS) describes the number of modes per energy.

    The phonon density of states (DOS) is a representation of the distribution of
    phonons in a material across different frequencies. It provides a histogram of the
    number of phonon states per frequency interval. Peaks and features in the DOS reveal
    the density of vibrational modes at specific frequencies. One can related these
    properties to study e.g. the heat capacity or the thermal conductivity.

    Projecting the phonon density of states (DOS) onto specific atoms highlights the
    contribution of each atomic species to the vibrational spectrum. This analysis helps
    to understand the role of individual elements' impact on the material's thermal and
    mechanical properties. The projected phonon DOS can guide towards engineering these
    properties by substitution of specific atoms. Additionally, the atom-specific
    projection allows for the identification of localized modes or vibrations associated
    with specific atomic species.
    """

    @_base.data_access
    def __str__(self):
        energies = self._raw_data.energies
        topology = self._topology()
        return f"""phonon DOS:
    [{energies[0]:0.2f}, {energies[-1]:0.2f}] mesh with {len(energies)} points
    {3 * topology.number_atoms()} modes
    {topology}"""

    @_base.data_access
    @documentation.format(selection=_phonon.selection_doc)
    def to_dict(self, selection=None):
        """Read the phonon DOS into a dictionary.

        Parameters
        ----------
        {selection}

        Returns
        -------
        dict
            Contains the energies at which the phonon DOS was computed. The total
            DOS is returned and any possible projected DOS selected by the *selection*
            argument.
        """
        return {
            "energies": self._raw_data.energies[:],
            "total": self._raw_data.dos[:],
            **self._read_data(selection),
        }

    @_base.data_access
    @documentation.format(selection=_phonon.selection_doc)
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

    def _read_data(self, selection):
        if not selection:
            return {}
        maps = {0: self._init_atom_dict(), 1: self._init_direction_dict()}
        selector = index.Selector(
            maps, self._raw_data.projections, use_number_labels=True
        )
        tree = select.Tree.from_selection(selection)
        return {
            selector.label(selection): selector[selection]
            for selection in tree.selections()
        }


def _series(data):
    energies = data["energies"]
    for name, dos in data.items():
        if name == "energies":
            continue
        yield graph.Series(energies, dos, name)
