# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy
import pathlib

from py4vasp import raw
from py4vasp._calculation import slice_
from py4vasp._calculation.dispatch import (
    DataSource,
    FileSource,
    merge_default,
    merge_graphs,
    merge_strings,
    quantity,
)
from py4vasp._raw.data_db import PairCorrelation_DB
from py4vasp._third_party import graph
from py4vasp._util import check, convert, documentation, index, select


def _selection_string(default):
    return f"""\
selection : str
    String specifying which pair-correlation functions are used. Select
    'total' for the total pair-correlation function or the name of any
    two ion types (e.g. 'Sr~Ti') for a specific pair-correlation function.
    When no selection is given, {default}. Separate
    distinct labels by commas or whitespace. For a complete list of all
    possible selections, please use

    >>> calculation.pair_correlation.labels()
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

    def read(self, selection=None) -> dict:
        return self.to_dict(selection)

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

    def to_database(self) -> dict:
        """Serialize pair-correlation data for database storage."""
        distance_min, distance_max = None, None
        if not check.is_none(self._raw_data.distances):
            distance_min = float(self._raw_data.distances[0])
            distance_max = float(self._raw_data.distances[-1])
        return {
            "pair_correlation": PairCorrelation_DB(
                distance_min=distance_min, distance_max=distance_max
            ),
        }

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
@documentation.format(examples=slice_.examples("pair_correlation", step="block"))
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

    {examples}
    """

    def __init__(self, source, quantity_name: str = "pair_correlation", steps=None):
        self._source = source
        self._quantity_name = quantity_name
        self._steps = steps

    @classmethod
    def from_data(cls, raw_pair_correlation: raw.PairCorrelation):
        """Create a PairCorrelation dispatcher from raw data."""
        return cls(source=DataSource(raw_pair_correlation))

    @classmethod
    def from_path(cls, path="."):
        """Create a PairCorrelation dispatcher from HDF5 files at *path*."""
        return cls(source=FileSource(path))

    @classmethod
    def from_file(cls, file_name):
        """Create a PairCorrelation dispatcher from a specific HDF5 file."""
        resolved = pathlib.Path(file_name).expanduser().resolve()
        return cls(source=FileSource(resolved.parent, file=file_name))

    @property
    def path(self):
        """Path used for file-export methods."""
        return self._source.path or pathlib.Path.cwd()

    @property
    def _path(self):
        return self.path

    def __getitem__(self, steps) -> "PairCorrelation":
        new = copy.copy(self)
        new._steps = steps
        return new

    def _handler_factory(self, raw_data):
        return PairCorrelationHandler.from_data(raw_data, steps=self._steps)

    @documentation.format(
        selection=_selection_string("all possibilities are read"),
        examples=slice_.examples("pair_correlation", "to_dict", "block"),
    )
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

        {examples}
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PairCorrelationHandler.read,
            selection,
        )

    @documentation.format(
        selection=_selection_string("all possibilities are read"),
        examples=slice_.examples("pair_correlation", "to_dict", "block"),
    )
    def to_dict(self, selection=None) -> dict:
        """Read the pair-correlation function and store it in a dictionary.

        Convenient alias for :py:meth:`read`.

        Parameters
        ----------
        {selection}

        Returns
        -------
        dict

        {examples}
        """
        return self.read(selection=selection)

    @documentation.format(
        selection=_selection_string("the total pair correlation is used"),
        examples=slice_.examples("pair_correlation", "to_graph", "block"),
    )
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

        {examples}
        """
        return merge_graphs(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PairCorrelationHandler.to_graph,
            selection,
        )

    def labels(self, selection: str | None = None) -> tuple:
        """Return all possible labels for the selection string."""
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PairCorrelationHandler.labels,
        )

    def __str__(self, selection: str | None = None) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PairCorrelationHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")


def _selection_string(default):
    return f"""\
selection : str
    String specifying which pair-correlation functions are used. Select
    'total' for the total pair-correlation function or the name of any
    two ion types (e.g. 'Sr~Ti') for a specific pair-correlation function.
    When no selection is given, {default}. Separate
    distinct labels by commas or whitespace. For a complete list of all
    possible selections, please use

    >>> calculation.pair_correlation.labels()
"""
