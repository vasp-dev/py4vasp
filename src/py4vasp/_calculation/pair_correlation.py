# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy

import numpy as np

from py4vasp import raw
from py4vasp._calculation.dispatch import (
    DataSource,
    _dispatch,
    merge_default,
    merge_graphs,
    merge_strings,
    merge_to_database,
    quantity,
)
from py4vasp._raw.models import PairCorrelationModel
from py4vasp._third_party import graph
from py4vasp._util import check, convert, documentation, index, select

# Minimum height of the total pair correlation function for a local maximum to be
# considered the first peak. The value corresponds to the ideal-gas baseline, so
# small bumps below it (e.g. numerical noise before the first shell) are ignored.
_FIRST_PEAK_THRESHOLD = 1.0


def _selection_string(default):
    return f"""\
selection : str
    String specifying which pair-correlation functions are used. Select
    'total' for the total pair-correlation function or the name of any
    two ion types (e.g. 'Sr~Ti') for a specific pair-correlation function.
    When no selection is given, {default}. Separate
    distinct labels by commas or whitespace. The :py:meth:`labels` method
    returns a complete list of all possible selections.
"""


class PairCorrelationHandler:
    """Handler for pair-correlation data — all data access and transformation."""

    def __init__(self, raw_pair_correlation: raw.PairCorrelation, steps=None):
        self._raw_data = raw_pair_correlation
        self._steps = steps

    @classmethod
    def from_data(
        cls, raw_pair_correlation: raw.PairCorrelation, steps=None
    ) -> "PairCorrelationHandler":
        return cls(raw_pair_correlation, steps)

    def __str__(self) -> str:
        distances = self._raw_data.distances
        pairs = ", ".join(self.labels())
        return f"""\
pair-correlation function:
    distances: [{distances[0]:0.2f}, {distances[-1]:0.2f}] {len(distances)} points
    pairs: {pairs}"""

    def to_dict(self, selection=None) -> dict:
        """Read the pair-correlation function and store it in a dictionary."""
        selection = self._default_selection_if_none(selection)
        return {
            "distances": self._raw_data.distances[:],
            **self._read_data(selection),
        }

    def to_graph(self, selection="total") -> graph.Graph:
        """Plot selected pair-correlation functions."""
        series = self._make_series(self.to_dict(selection))
        return graph.Graph(series, xlabel="Distance (Å)", ylabel="Pair correlation")

    def labels(self) -> tuple:
        """Return all possible labels for the selection string."""
        return tuple(convert.text_to_string(label) for label in self._raw_data.labels)

    def to_database(self) -> PairCorrelationModel:
        """Serialize pair-correlation data for database storage."""
        distance_min, distance_max = None, None
        if not check.is_none(self._raw_data.distances):
            distance_min = float(self._raw_data.distances[0])
            distance_max = float(self._raw_data.distances[-1])
        first_peak_position, first_peak_height = self._first_peak()
        return PairCorrelationModel(
            distance_min=distance_min,
            distance_max=distance_max,
            first_peak_position=first_peak_position,
            first_peak_height=first_peak_height,
        )

    def _first_peak(self):
        """Position and height of the first peak of the total pair correlation.

        Scans the existing sample points for the first strict local maximum of the
        total g(r) whose height exceeds :data:`_FIRST_PEAK_THRESHOLD`, so small
        bumps before the first real shell are skipped. Returns ``(None, None)`` when
        no such peak exists.
        """
        if check.is_none(self._raw_data.distances) or check.is_none(
            self._raw_data.function
        ):
            return None, None
        distances = np.asarray(self._raw_data.distances[:])
        # index 0 of the function is the total pair correlation by convention
        total = np.asarray(self._raw_data.function)[self._steps_or_last, 0]
        if distances.shape != total.shape or total.size < 3:
            return None, None
        for i in range(1, total.size - 1):
            is_local_max = total[i] > total[i - 1] and total[i] > total[i + 1]
            if is_local_max and total[i] > _FIRST_PEAK_THRESHOLD:
                return float(distances[i]), float(total[i])
        return None, None

    @property
    def _steps_or_last(self):
        return -1 if self._steps is None else self._steps

    def _default_selection_if_none(self, selection):
        return selection or ",".join(self.labels())

    def _read_data(self, selection):
        map_ = {1: self._init_pair_correlation_dict()}
        selector = index.Selector(map_, self._raw_data.function)
        tree = select.Tree.from_selection(selection)
        return {
            selector.label(selection): selector[selection][self._steps_or_last]
            for selection in tree.selections()
        }

    def _init_pair_correlation_dict(self):
        return {label: i for i, label in enumerate(self.labels())}

    def _make_series(self, selected_data):
        distances = selected_data["distances"]
        return [
            graph.Series(x=distances, y=data, label=label)
            for label, data in selected_data.items()
            if label != "distances"
        ]


@quantity("pair_correlation")
class PairCorrelation(graph.Mixin):
    """The pair-correlation function measures the distribution of atoms.

    A pair-correlation function is a statistical measure to describe the spatial
    distribution of atoms within a system. Specifically, the pair correlation
    function quantifies the probability density of finding two particles at specific
    separation distances. This function is helpful in the study of liquids and solids
    because it acts as a fingerprint of the system that can be compared to
    X-ray or neutron scattering experiments. Another use case is the detection
    of specific phases.

    Use this class to inspect the pair-correlation function computed by VASP for
    all pairs of ionic types. You can control how often VASP samples the pair
    correlation function with the :tag:`NBLOCK` tag. If you want to split your
    trajectory into multiple subsets include the tag :tag:`KBLOCK` in your INCAR
    file.

    Examples
    --------
    First, we create some example data so that you can follow along. Please define a
    variable `path` with the path to a directory that does not contain any VASP
    calculation data. Alternatively, use your own data if you have run VASP.

    >>> from py4vasp import demo
    >>> calculation = demo.calculation(path)

    Plot the total pair-correlation function of the final block

    >>> calculation.pair_correlation.plot()
    Graph(series=[Series(x=array([...]), y=array([...]), label='total', ...)], ...)

    A summary of the pairs the function resolves is printed by

    >>> print(calculation.pair_correlation)
    pair-correlation function:
        distances: [0.00, 8.00] 301 points
        pairs: total, Sr~Sr, Sr~Ti, Sr~O, Ti~Ti, Ti~O, O~O

    Use the [] operator to select the blocks VASP sampled

    >>> calculation.pair_correlation[:].read()["total"].shape
    (12, 301)
    """

    def __init__(self, source, quantity_name: str = "pair_correlation", steps=None):
        self._source = source
        self._quantity_name = quantity_name
        self._steps = steps

    @classmethod
    def from_data(cls, raw_pair_correlation: raw.PairCorrelation):
        """Create a PairCorrelation dispatcher from raw data."""
        return cls(source=DataSource(raw_pair_correlation))

    def __getitem__(self, steps) -> "PairCorrelation":
        new = copy.copy(self)
        new._steps = steps
        return new

    def _handler_factory(self, raw_data):
        return PairCorrelationHandler.from_data(raw_data, steps=self._steps)

    @documentation.format(selection=_selection_string("all possibilities are read"))
    def read(self, selection=None) -> dict:
        """Read the pair-correlation function and store it in a dictionary.

        Parameters
        ----------
        {selection}

        Returns
        -------
        dict
            Contains the labels corresponding to the selection and the associated
            pair-correlation function for every selected block. Furthermore, the
            dictionary contains the distances at which the pair-correlation functions
            are evaluated.

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not contain any VASP
        calculation data. Alternatively, use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        Without a selection you obtain the distances and every pair of the crystal

        >>> sorted(calculation.pair_correlation.read())
        ['O~O', 'Sr~O', 'Sr~Sr', 'Sr~Ti', 'Ti~O', 'Ti~Ti', 'distances', 'total']

        Select a single pair to see where its neighbours are. The nearest neighbour of
        titanium is an oxygen roughly two Angstrom away, so the function rises there

        >>> data = calculation.pair_correlation.read("Ti~O")
        >>> round(data["distances"][data["Ti~O"].argmax()], 2)
        np.float64(1.97)
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PairCorrelationHandler.to_dict,
        )

    def to_dict(self, selection=None) -> dict:
        """Convenient alias for :py:meth:`read`."""
        return self.read(selection=selection)

    @documentation.format(selection=_selection_string("the total one is used"))
    def to_graph(self, selection="total") -> graph.Graph:
        """Plot selected pair-correlation functions.

        Parameters
        ----------
        {selection}

        Returns
        -------
        Graph
            The graph plots the pair-correlation function for all selected blocks
            and ion pairs.

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not contain any VASP
        calculation data. Alternatively, use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        >>> calculation.pair_correlation.to_graph()
        Graph(series=[Series(x=array([...]), y=array([...]), label='total', ...)],
              xlabel='Distance (Å)', ...)

        Compare the neighbours of two pairs by selecting both

        >>> calculation.pair_correlation.to_graph("Ti~O, Sr~O")
        Graph(series=[Series(..., label='Ti~O', ...), Series(..., label='Sr~O', ...)], ...)
        """
        return merge_graphs(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PairCorrelationHandler.to_graph,
        )

    def labels(self) -> tuple:
        """Return all possible labels for the selection string.

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not contain any VASP
        calculation data. Alternatively, use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        >>> calculation.pair_correlation.labels()
        ('total', 'Sr~Sr', 'Sr~Ti', 'Sr~O', 'Ti~Ti', 'Ti~O', 'O~O')
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            PairCorrelationHandler.labels,
        )

    def print(self, selection: str | None = None) -> None:
        """Print a string representation of this quantity.

        Parameters
        ----------
        selection : str | None
            Select which source of the quantity is printed. If you select multiple
            sources, py4vasp prints one block per source.
        """
        print(self.__str__(selection))

    def selections(self) -> dict:
        """Returns possible alternatives for this particular quantity VASP can produce.

        The returned dictionary contains a single item with the name of the quantity
        mapping to all possible selections. Each of these selections may be passed to
        the other methods of this quantity to choose which output of VASP is used.

        Returns
        -------
        dict
            The key indicates this quantity and the value lists the possible choices
            for the selection argument of its other methods.
        """
        from py4vasp._raw import definition as raw_module

        return {self._quantity_name: list(raw_module.selections(self._quantity_name))}

    def __str__(self, selection=None) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PairCorrelationHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")

    def _to_database(self) -> dict:
        """Return {quantity[_selection]: handler_result} for database storage."""
        return merge_to_database(
            self._source,
            self._quantity_name,
            PairCorrelationHandler.from_data,
            PairCorrelationHandler.to_database,
        )
