# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pathlib

import numpy as np

from py4vasp import raw
from py4vasp._calculation import phonon
from py4vasp._calculation._dispersion import DispersionHandler
from py4vasp._calculation._stoichiometry import StoichiometryHandler
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    merge_graphs,
    merge_strings,
    quantity,
)
from py4vasp._third_party import graph
from py4vasp._util import convert, database, documentation, index, select


class PhononBandHandler:
    """Handler for phonon band structure data."""

    def __init__(self, raw_phonon_band: raw.PhononBand):
        self._raw_phonon_band = raw_phonon_band

    @classmethod
    def from_data(cls, raw_phonon_band: raw.PhononBand) -> "PhononBandHandler":
        return cls(raw_phonon_band)

    def __str__(self) -> str:
        return f"""phonon band data:
    {self._raw_phonon_band.dispersion.eigenvalues.shape[0]} q-points
    {self._raw_phonon_band.dispersion.eigenvalues.shape[1]} modes
    {self._stoichiometry()}"""

    def to_dict(self) -> dict:
        dispersion = self._dispersion().to_dict()
        return {
            "qpoint_distances": dispersion["kpoint_distances"],
            "qpoint_labels": dispersion.get("kpoint_labels"),
            "bands": dispersion["eigenvalues"],
            "modes": self._modes(),
        }

    def to_database(self) -> dict:
        stoichiometry = self._stoichiometry().to_database()
        dispersion = self._dispersion().to_database()
        return database.combine_db_dicts(
            {"phonon_band": {}},
            stoichiometry,
            dispersion,
        )

    def to_graph(self, selection=None, width=1.0) -> graph.Graph:
        projections = self._projections(selection, width)
        g = self._dispersion().plot(projections)
        g.ylabel = "ω (THz)"
        return g

    def selections(self) -> dict:
        atoms = self._init_atom_dict().keys()
        return {
            "atom": sorted(atoms, key=self._sort_key),
            "direction": ["x", "y", "z"],
        }

    def _dispersion(self) -> DispersionHandler:
        return DispersionHandler.from_data(self._raw_phonon_band.dispersion)

    def _stoichiometry(self) -> StoichiometryHandler:
        return StoichiometryHandler.from_data(self._raw_phonon_band.stoichiometry)

    def _modes(self) -> np.ndarray:
        return convert.to_complex(self._raw_phonon_band.eigenvectors[:])

    def _projections(self, selection, width):
        if not selection:
            return None
        maps = {2: self._init_atom_dict(), 3: self._init_direction_dict()}
        selector = index.Selector(maps, np.abs(self._modes()), use_number_labels=True)
        tree = select.Tree.from_selection(selection)
        return {selector.label(sel): width * selector[sel] for sel in tree.selections()}

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


@quantity("band", group="phonon")
class PhononBand(graph.Mixin):
    """The phonon band structure contains the **q**-resolved phonon eigenvalues.

    The phonon band structure is a graphical representation of the phonons. It
    illustrates the relationship between the frequency of modes and their corresponding
    wave vectors in the Brillouin zone. Each line or branch in the band structure
    represents a specific phonon, and the slope of these branches provides information
    about their velocity.

    The phonon band structure includes the dispersion relations of phonons, which reveal
    how vibrational frequencies vary with direction in the crystal lattice. The presence
    of band gaps or band crossings indicates the material's ability to conduct or
    insulate heat. Additionally, the branches near the high-symmetry points in the
    Brillouin zone offer insights into the material's anharmonicity and thermal
    conductivity. Furthermore, phonons with imaginary frequencies indicate the presence
    of a structural instability.
    """

    def __init__(self, source, quantity_name: str = "phonon_band"):
        self._source = source
        self._quantity_name = quantity_name
        self._path = pathlib.Path.cwd()

    @classmethod
    def from_data(cls, raw_phonon_band: raw.PhononBand) -> "PhononBand":
        return cls(source=DataSource(raw_phonon_band))

    def _handler_factory(self, raw):
        return PhononBandHandler.from_data(raw)

    def __str__(self, selection=None) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PhononBandHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self, selection=None) -> dict:
        """Read the phonon band structure into a dictionary.

        Returns
        -------
        dict
            Contains the **q**-point path for plotting phonon band structures and
            the phonon bands. In addition the phonon modes are returned.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PhononBandHandler.to_dict,
        )

    def to_dict(self, selection=None) -> dict:
        """Convenient alias for :py:meth:`read`."""
        return self.read(selection=selection)

    @documentation.format(selection=phonon.selection_doc)
    def to_graph(self, selection: str | None = None, width: float = 1.0) -> graph.Graph:
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
        return merge_graphs(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PhononBandHandler.to_graph,
            width,
        )

    def selections(self, selection=None) -> dict:
        """Return atom and direction selections available for projection."""
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PhononBandHandler.selections,
        )
