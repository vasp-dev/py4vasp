# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pathlib

from py4vasp import raw
from py4vasp._calculation._stoichiometry import StoichiometryHandler
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    merge_graphs,
    merge_strings,
    quantity,
)
from py4vasp._raw.data_db import PhononDos_DB
from py4vasp._third_party import graph
from py4vasp._util import check, documentation, index, select
from py4vasp._calculation import phonon


class PhononDosHandler:
    """Handler for phonon DOS data."""

    def __init__(self, raw_phonon_dos: raw.PhononDos):
        self._raw_phonon_dos = raw_phonon_dos

    @classmethod
    def from_data(cls, raw_phonon_dos: raw.PhononDos) -> "PhononDosHandler":
        return cls(raw_phonon_dos)

    def __str__(self) -> str:
        energies = self._raw_phonon_dos.energies
        stoichiometry = self._stoichiometry()
        return f"""phonon DOS:
    [{energies[0]:0.2f}, {energies[-1]:0.2f}] mesh with {len(energies)} points
    {3 * stoichiometry.number_atoms()} modes
    {stoichiometry}"""

    def to_dict(self, selection=None) -> dict:
        """Read the phonon DOS into a dictionary.

        Parameters
        ----------
        selection : str
            A string specifying the projection of the phonon modes onto atoms and
            directions. See `selections` for available options.

        Returns
        -------
        dict
            Contains the energies at which the phonon DOS was computed. The total
            DOS is returned and any possible projected DOS selected by the *selection*
            argument.
        """
        return {
            "energies": self._raw_phonon_dos.energies[:],
            "total": self._raw_phonon_dos.dos[:],
            **self._read_data(selection),
        }

    def to_database(self) -> dict:
        energy_min = (
            float(self._raw_phonon_dos.energies[0])
            if not check.is_none(self._raw_phonon_dos.energies)
            else None
        )
        energy_max = (
            float(self._raw_phonon_dos.energies[-1])
            if not check.is_none(self._raw_phonon_dos.energies)
            else None
        )
        return {
            "phonon_dos": PhononDos_DB(energy_min=energy_min, energy_max=energy_max),
        }

    def to_graph(self, selection=None) -> graph.Graph:
        data = self.to_dict(selection)
        return graph.Graph(
            series=list(_series(data)),
            xlabel="ω (THz)",
            ylabel="DOS (1/THz)",
        )

    def selections(self) -> dict:
        atoms = self._init_atom_dict().keys()
        return {
            "atom": sorted(atoms, key=self._sort_key),
            "direction": ["x", "y", "z"],
        }

    def _stoichiometry(self) -> StoichiometryHandler:
        return StoichiometryHandler.from_data(self._raw_phonon_dos.stoichiometry)

    def _read_data(self, selection) -> dict:
        if not selection:
            return {}
        maps = {0: self._init_atom_dict(), 1: self._init_direction_dict()}
        selector = index.Selector(
            maps, self._raw_phonon_dos.projections, use_number_labels=True
        )
        tree = select.Tree.from_selection(selection)
        return {selector.label(sel): selector[sel] for sel in tree.selections()}

    def _init_atom_dict(self) -> dict:
        return {
            key: value.indices
            for key, value in self._stoichiometry().read().items()
            if key != select.all
        }

    def _init_direction_dict(self) -> dict:
        return {
            "x": slice(0, 1),
            "y": slice(1, 2),
            "z": slice(2, 3),
        }

    def _sort_key(self, key) -> bool:
        return key.isdecimal()


@quantity("dos", group="phonon")
class PhononDos(graph.Mixin):
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

    def __init__(self, source, quantity_name: str = "phonon_dos"):
        self._source = source
        self._quantity_name = quantity_name
        self._path = pathlib.Path.cwd()

    @classmethod
    def from_data(cls, raw_phonon_dos: raw.PhononDos) -> "PhononDos":
        return cls(source=DataSource(raw_phonon_dos))

    def _handler_factory(self, raw):
        return PhononDosHandler.from_data(raw)

    def __str__(self, selection=None) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PhononDosHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    @documentation.format(selection=phonon.selection_doc)
    def read(self, selection: str | None = None) -> dict:
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
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PhononDosHandler.to_dict,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Convenient alias for :py:meth:`read`. Please read the documentation there."""
        return self.read(selection=selection)

    @documentation.format(selection=phonon.selection_doc)
    def to_graph(self, selection: str | None = None) -> graph.Graph:
        """Generate a graph of the selected phonon DOS.

        Parameters
        ----------
        {selection}

        Returns
        -------
        Graph
            The graph contains the total DOS. If a selection is given, in addition the
            projected DOS is shown.
        """
        return merge_graphs(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PhononDosHandler.to_graph,
        )

    def selections(self, selection=None) -> dict:
        """Return atom and direction selections available for projection."""
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PhononDosHandler.selections,
        )


def _series(data):
    energies = data["energies"]
    for name, dos in data.items():
        if name == "energies":
            continue
        yield graph.Series(energies, dos, name)
