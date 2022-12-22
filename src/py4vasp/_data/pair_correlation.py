# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import exception
from py4vasp._data import base, slice_
from py4vasp._third_party import graph
from py4vasp._util import convert, documentation, select


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


@documentation.format(examples=slice_.examples("pair_correlation", step="block"))
class PairCorrelation(slice_.Mixin, base.Refinery, graph.Mixin):
    """The pair-correlation function for one or several blocks of an MD simulation.

    Use this class to inspect how the correlation of the position of different
    ions types in an MD simulation. The pair-correlation function gives insight
    into the structural properties and may help to identify certain orders in
    the system.

    {examples}
    """

    @base.data_access
    @documentation.format(
        selection=_selection_string("all possibilities are read"),
        examples=slice_.examples("pair_correlation", "to_dict", "block"),
    )
    def to_dict(self, selection=select.all):
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
        return {
            "distances": self._raw_data.distances[:],
            **self._read_data(selection),
        }

    @base.data_access
    @documentation.format(
        selection=_selection_string("the total pair correlation is used"),
        examples=slice_.examples("pair_correlation", "to_graph", "block"),
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

    @base.data_access
    def labels(self):
        "Return all possible labels for the selection string."
        return tuple(convert.text_to_string(label) for label in self._raw_data.labels)

    def _read_data(self, selection):
        return {
            label: self._raw_data.function[self._steps, index]
            for label, index in self._parse_user_selection(selection)
        }

    def _parse_user_selection(self, selection):
        if selection == select.all:
            return self._select_all()
        else:
            return self._select_specified_subset(selection)

    def _select_all(self):
        for index, label in enumerate(self.labels()):
            yield label, index

    def _select_specified_subset(self, selection):
        tree = select.Tree.from_selection(selection)
        labels = self.labels()
        for node in tree.nodes:
            yield self._find_matching_label(node.content, labels)

    def _find_matching_label(self, content, labels):
        for index, label in enumerate(labels):
            if self._content_matches_label(content, label):
                return label, index
        labels = ", ".join(labels)
        message = (
            f"{content} is not a valid label. Please check for possible spelling errors. "
            f"The following labels are possible: {labels}."
        )
        raise exception.IncorrectUsage(message)

    def _content_matches_label(self, content, label):
        if isinstance(content, str):
            return content == label
        else:
            reversed_content = content.separator.join(reversed(content.group))
            return str(content) == label or reversed_content == label

    def _make_series(self, selected_data):
        distances = selected_data["distances"]
        return [
            graph.Series(x=distances, y=data, name=label)
            for label, data in selected_data.items()
            if label != "distances"
        ]
