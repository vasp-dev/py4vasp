# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp.data import _base, _slice, _export
import py4vasp.exceptions as exception
import py4vasp._third_party.graph as _graph
import py4vasp._util.convert as _convert
import py4vasp._util.documentation as _documentation
import py4vasp._util.selection as _selection

_pair_correlation_docs = f"""
The pair-correlation function for one or several blocks of an MD simulation.

Use this class to inspect how the correlation of the position of different
ions types in an MD simulation. The pair-correlation function gives insight
into the structural properties and may help to identify certain orders in
the system.

{_slice.examples("pair_correlation", step="block")}
""".strip()


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


_read_docs = f"""Read the pair-correlation function and store it in a dictionary.

Parameters
----------
{_selection_string("all possibilities are read")}

Returns
-------
dict
    Contains the labels corresponding to the selection and the associated
    pair-correlation function for every selected block. Furthermore, the
    dictionary contains the distances at which the pair-correlation functions
    are evaluated.

{_slice.examples("pair_correlation", "read", "block")}"""

_plot_docs = f"""Plot selected pair-correlation functions.

Parameters
----------
{_selection_string("the total pair correlation is used")}

Returns
-------
Graph
    The graph plots the pair-correlation function for all selected blocks
    and ion pairs. Note that the various blocks with the same legend and
    only different ion combinations use different color schemes.

{_slice.examples("pair_correlation", "plot", "block")}"""

_to_plotly_docs = f"""Plot selected pair-correlation functions.

Parameters
----------
{_selection_string("the total pair correlation is used")}

Returns
-------
plotly.graph_objects.Figure
    plotly figure containing the pair correlation for every selected blocks.

{_slice.examples("pair_correlation", "to_plotly", "block")}"""


@_documentation.add(_pair_correlation_docs)
class PairCorrelation(_slice.Mixin, _base.Refinery, _export.Image):
    @_base.data_access
    @_documentation.add(_read_docs)
    def to_dict(self, selection=_selection.all):
        return {
            "distances": self._raw_data.distances[:],
            **self._read_data(selection),
        }

    @_base.data_access
    @_documentation.add(_plot_docs)
    def plot(self, selection="total"):
        series = self._make_series(self.to_dict(selection))
        return _graph.Graph(series, xlabel="Distance (Å)", ylabel="Pair correlation")

    @_base.data_access
    @_documentation.add(_to_plotly_docs)
    def to_plotly(self, selection="total"):
        return self.plot(selection).to_plotly()

    @_base.data_access
    def labels(self):
        "Return all possible labels for the selection string."
        return tuple(_convert.text_to_string(label) for label in self._raw_data.labels)

    def _read_data(self, selection):
        return {
            label: self._raw_data.function[self._steps, index]
            for label, index in self._parse_selection(selection)
        }

    def _parse_selection(self, selection):
        if selection == _selection.all:
            return self._select_all()
        else:
            return self._select_specified_subset(selection)

    def _select_all(self):
        for index, label in enumerate(self.labels()):
            yield label, index

    def _select_specified_subset(self, selection):
        tree = _selection.Tree.from_selection(selection)
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
            _graph.Series(x=distances, y=data, name=label)
            for label, data in selected_data.items()
            if label != "distances"
        ]
