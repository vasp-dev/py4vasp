# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp.calculation import _base
from py4vasp._third_party import graph
from py4vasp._util import convert, documentation, index, select
from py4vasp.calculation import _slice


def _selection_string(default):
    return f"""\
selection : str
    String specifying which pair-correlation functions are used. Select
    'total' for the total pair-correlation function or the name of any
    two ion types (e.g. 'Sr~Ti') for a specific pair-correlation function.
    When no selection is given, {default}. Separate
    distinct labels by commas or whitespace. For a complete list of all
    possible selections, please use

    >>> calc.pair_correlation.labels()
"""


@documentation.format(examples=_slice.examples("pair_correlation", step="block"))
class PairCorrelation(_slice.Mixin, _base.Refinery, graph.Mixin):
    """The pair-correlation function for one or several blocks of an MD simulation.

    Use this class to inspect how the correlation of the position of different
    ions types in an MD simulation. The pair-correlation function gives insight
    into the structural properties and may help to identify certain orders in
    the system.

    {examples}
    """

    @_base.data_access
    @documentation.format(
        selection=_selection_string("all possibilities are read"),
        examples=_slice.examples("pair_correlation", "to_dict", "block"),
    )
    def to_dict(self, selection=None):
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
        selection = self._default_selection_if_none(selection)
        return {
            "distances": self._raw_data.distances[:],
            **self._read_data(selection),
        }

    @_base.data_access
    @documentation.format(
        selection=_selection_string("the total pair correlation is used"),
        examples=_slice.examples("pair_correlation", "to_graph", "block"),
    )
    def to_graph(self, selection="total"):
        """Plot selected pair-correlation functions.

        Parameters
        ----------
        {selection}

        Returns
        -------
        Graph
            The graph plots the pair-correlation function for all selected blocks
            and ion pairs. Note that the various blocks with the same legend and
            only different ion combinations use different color schemes.

        {examples}
        """
        series = self._make_series(self.to_dict(selection))
        return graph.Graph(series, xlabel="Distance (Å)", ylabel="Pair correlation")

    @_base.data_access
    def labels(self):
        "Return all possible labels for the selection string."
        return tuple(convert.text_to_string(label) for label in self._raw_data.labels)

    def _default_selection_if_none(self, selection):
        return selection or ",".join(self.labels())

    def _read_data(self, selection):
        map_ = {1: self._init_pair_correlation_dict()}
        selector = index.Selector(map_, self._raw_data.function)
        tree = select.Tree.from_selection(selection)
        return {
            selector.label(selection): selector[selection][self._steps]
            for selection in tree.selections()
        }

    def _init_pair_correlation_dict(self):
        return {label: i for i, label in enumerate(self.labels())}

    def _make_series(self, selected_data):
        distances = selected_data["distances"]
        return [
            graph.Series(x=distances, y=data, name=label)
            for label, data in selected_data.items()
            if label != "distances"
        ]
