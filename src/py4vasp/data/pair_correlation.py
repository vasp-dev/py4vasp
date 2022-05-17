from py4vasp.data import _base, _trajectory
import py4vasp.exceptions as exception
import py4vasp._third_party.graph as _graph
import py4vasp._util.convert as _convert
import py4vasp._util.selection as _selection


class PairCorrelation(_trajectory.DataTrajectory):
    "TODO"
    read = _base.RefinementDescriptor("_to_dict")
    plot = _base.RefinementDescriptor("_to_plotly")

    def _to_dict(self, selection=_selection.all):
        return {
            "distances": self._raw_data.distances[:],
            **self._read_data(selection),
        }

    def _to_plotly(self, selection="total"):
        series = self._make_series(self._to_dict(selection))
        return _graph.Graph(series, xlabel="Distance (Å)", ylabel="Pair correlation")

    def _read_data(self, selection):
        return {
            label: self._raw_data.function[self._steps, index]
            for label, index in self._parse_selection(selection)
        }

    def _parse_selection(self, selection):
        labels = [_convert.text_to_string(label) for label in self._raw_data.labels]
        if selection == _selection.all:
            return self._select_all(labels)
        else:
            return self._select_specified_subset(selection, labels)

    def _select_all(self, labels):
        for index, label in enumerate(labels):
            yield label, index

    def _select_specified_subset(self, selection, labels):
        tree = _selection.Tree.from_selection(selection)
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
