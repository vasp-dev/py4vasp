# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pathlib

import numpy as np

from py4vasp import raw
from py4vasp._calculation._dispersion import DispersionHandler
from py4vasp._calculation._stoichiometry import StoichiometryHandler
from py4vasp._calculation.dispatch import DataSource, merge_default, merge_graphs, merge_strings, quantity
from py4vasp._third_party import graph
from py4vasp._util import convert, database, index, select


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

    def read(self) -> dict:
        return self.to_dict()

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
        return {
            selector.label(sel): width * selector[sel]
            for sel in tree.selections()
        }

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
    """The phonon band structure contains the **q**-resolved phonon eigenvalues."""

    def __init__(self, source, quantity_name: str = "phonon_band"):
        self._source = source
        self._quantity_name = quantity_name
        self._path = pathlib.Path.cwd()

    @classmethod
    def from_data(cls, raw_phonon_band: raw.PhononBand) -> "PhononBand":
        return cls(source=DataSource(raw_phonon_band))

    def _handler_factory(self, raw):
        return PhononBandHandler.from_data(raw)

    def __str__(self) -> str:
        return merge_strings(
            self._source, self._quantity_name, None,
            self._handler_factory, PhononBandHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self, selection: str | None = None) -> dict:
        return merge_default(
            self._source, self._quantity_name, selection,
            self._handler_factory, PhononBandHandler.read,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Read the phonon band structure into a dictionary."""
        return self.read(selection=selection)

    def to_graph(self, selection: str | None = None, width: float = 1.0) -> graph.Graph:
        """Generate a graph of the phonon bands."""
        return merge_graphs(
            self._source, self._quantity_name, None,
            self._handler_factory, PhononBandHandler.to_graph,
            selection, width,
        )

    def selections(self, selection: str | None = None) -> dict:
        """Return atom and direction selections available for projection."""
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, PhononBandHandler.selections,
        )
