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
    _dispatch,
    merge_default,
    merge_graphs,
    merge_strings,
    merge_to_database,
    quantity,
)
from py4vasp._raw.models import PhononBandModel
from py4vasp._third_party import graph
from py4vasp._util import convert, documentation, index, select


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
            "bands": self._energies(dispersion["eigenvalues"]),
            "modes": self._modes(),
        }

    def to_database(self) -> PhononBandModel:
        dispersion = self._dispersion().to_database()
        return PhononBandModel(
            eigenvalue_min=self._energies(dispersion.eigenvalue_min),
            eigenvalue_max=self._energies(dispersion.eigenvalue_max),
        )

    def to_graph(self, selection=None, width=1.0) -> graph.Graph:
        # the dispersion draws VASP's THz branches; a phonon spectrum is a few tens of
        # meV wide, so eV would compress every tick into three leading zeros
        projections = self._projections(selection, width)
        g = self._dispersion().plot(projections)
        for series in g.series:
            series.y = self._energies(series.y) * convert.EV_TO_MEV
        g.ylabel = "ω (meV)"
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

    def _energies(self, frequencies):
        # VASP reports the branches in THz. The conversion belongs here and not in
        # DispersionHandler, which the electronic band shares and which is already eV.
        # An unstable mode stays negative so that its branch is drawn below zero;
        # PhononMode reports the same mode as an imaginary energy instead.
        return frequencies / convert.EV_TO_THZ

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
    insulate heat. :py:meth:`read` reports every branch as the energy ħω in eV, while
    everything drawn or exported from the graph -- :py:meth:`plot`, ``to_plotly``,
    ``to_frame``, ``to_csv`` -- uses meV, so a csv written from this quantity is in meV
    even though ``read`` gave you eV. Additionally, the branches near the
    Brillouin zone offer insights into the material's anharmonicity and thermal
    conductivity. Furthermore, phonons with imaginary frequencies indicate the presence
    of a structural instability.

    See Also
    --------
    py4vasp._calculation.phonon_mode.PhononMode :
        Animates the modes behind these branches. Select its "dispersion" source to
        watch how the atoms move at a particular **q** point of this path.
    """

    def __init__(self, source, quantity_name: str = "phonon_band"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_phonon_band: raw.PhononBand) -> "PhononBand":
        return cls(source=DataSource(raw_phonon_band))

    def _handler_factory(self, raw):
        return PhononBandHandler.from_data(raw)

    def print(self, selection: str | None = None) -> None:
        """Print a string representation of this quantity.

        Parameters
        ----------
        selection : str | None
            Select which source of the quantity is printed. If you select multiple
            sources, py4vasp prints one block per source.
        """
        print(self.__str__(selection))

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
            the phonon bands as the energy ħω in eV. In addition the phonon modes
            are returned.

        Notes
        -----
        VASP reports the branches in THz; py4vasp converts them to an energy so that
        every quantity speaks the same unit. An unstable mode comes back as a negative
        energy, which is how its branch is conventionally drawn. The same modes are
        available from :py:class:`~py4vasp._calculation.phonon_mode.PhononMode`, which
        reports an unstable mode as an imaginary energy instead.

        Examples
        --------
        First, we create some example data so that you can follow along. Alternatively,
        use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation()

        The bands are resolved by **q** point and by mode

        >>> band = calculation.phonon.band.read()
        >>> band["bands"].shape
        (164, 21)

        A crystal of oxides vibrates within the first hundred meV, so the energies are
        small numbers when expressed in eV

        >>> round(float(band["bands"].max()), 3)
        0.08
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
            Contains the phonon band structure for all the **q** points, drawn in meV.
            If a selection is provided, the width of the bands is adjusted according to
            the projection.

        Examples
        --------
        First, we create some example data so that you can follow along. Alternatively,
        use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation()

        The graph is drawn in meV, where a phonon spectrum reads naturally, whereas
        :py:meth:`read` reports the same energies in eV

        >>> calculation.phonon.band.to_graph()
        Graph(series=[Series(..., label='bands', ...)], ..., ylabel='ω (meV)', ...)

        Widen a branch by how much the selected atoms contribute to it

        >>> calculation.phonon.band.to_graph("Sr, Ti, O")
        Graph(series=[Series(..., label='Sr', ...), Series(..., label='Ti', ...),
              Series(..., label='O', ...)], ...)
        """
        return merge_graphs(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PhononBandHandler.to_graph,
            width=width,
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

    def _to_database(self) -> dict:
        """Return {quantity[_selection]: handler_result} for database storage."""
        return merge_to_database(
            self._source,
            self._quantity_name,
            PhononBandHandler.from_data,
            PhononBandHandler.to_database,
        )
