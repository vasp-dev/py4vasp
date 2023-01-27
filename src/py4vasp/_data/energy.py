# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import exception
from py4vasp._data import base, slice_
from py4vasp._third_party import graph
from py4vasp._util import check, convert, documentation, select


def _selection_string(default):
    return f"""selection : str or None
    String specifying the labels of the energy to be read. A substring
    of the label is sufficient. If no energy is select this will default
    to selecting {default}. Separate distinct labels by commas. For a
    complete list of all possible selections, please use
    >>> calc.energy.labels()
"""


@documentation.format(examples=slice_.examples("energy"))
class Energy(slice_.Mixin, base.Refinery, graph.Mixin):
    """The energy data for one or several steps of a relaxation or MD simulation.

    You can use this class to inspect how the ionic relaxation converges or
    during an MD simulation whether the total energy is conserved.

    {examples}
    """

    @base.data_access
    def __str__(self):
        text = f"Energies at {self._step_string()}:"
        values = self._raw_data.values[self._last_step_in_slice]
        for label, value in zip(self._raw_data.labels, values):
            label = f"{convert.text_to_string(label):22.22}"
            text += f"\n   {label}={value:17.6f}"
        return text

    def _step_string(self):
        if self._is_slice:
            range_ = range(len(self._raw_data.values))[self._slice]
            start = range_.start + 1  # convert to Fortran index
            stop = range_.stop
            return f"step {stop} of range {start}-{stop}"
        elif self._steps == -1:
            return "final step"
        else:
            return f"step {self._steps + 1}"

    @base.data_access
    @documentation.format(
        selection=_selection_string("all energies"),
        examples=slice_.examples("energy", "to_dict"),
    )
    def to_dict(self, selection=select.all):
        """Read the energy data and store it in a dictionary.

        Parameters
        ----------
        {selection}

        Returns
        -------
        dict
            Contains the exact labels corresponding to the selection and the
            associated energies for every selected ionic step.

        {examples}
        """
        return {
            label: self._raw_data.values[self._steps, index]
            for label, index in self._parse_user_selection(selection)
        }

    @base.data_access
    @documentation.format(
        selection=_selection_string("the total energy"),
        examples=slice_.examples("energy", "to_graph"),
    )
    def to_graph(self, selection="TOTEN"):
        """Read the energy data and generate a figure of the selected components.

        Parameters
        ----------
        {selection}

        Returns
        -------
        Graph
            figure containing the selected energies for every selected ionic step.

        {examples}
        """
        yaxes = self._create_yaxes(selection)
        return graph.Graph(
            series=self._make_series(yaxes, selection),
            xlabel="Step",
            ylabel=yaxes.ylabel,
            y2label=yaxes.y2label,
        )

    @base.data_access
    @documentation.format(
        selection=_selection_string("the total energy"),
        examples=slice_.examples("energy", "to_numpy"),
    )
    def to_numpy(self, selection="TOTEN"):
        """Read the energy of the selected steps.

        Parameters
        ----------
        {selection}

        Returns
        -------
        float or np.ndarray or tuple
            Contains energies associated with the selection for the selected ionic step(s).
            When only a single step is inquired, result is a float otherwise an array.
            If you select multiple quantities a tuple of them is returned.

        {examples}
        """
        result = tuple(
            self._raw_data.values[self._steps, index]
            for _, index in self._parse_user_selection(selection)
        )
        return _unpack_if_only_one_element(result)

    @base.data_access
    def labels(self, selection=select.all):
        "Return the labels corresponding to a particular selection defaulting to all labels."
        return [label for label, _ in self._parse_user_selection(selection)]

    def _parse_user_selection(self, selection):
        # NOTE it would be nice to use SelectionTree instead, however that requires
        # the labels may have spaces so it might lead to redunandancies
        indices = self._find_selection_indices(selection)
        get_label = lambda index: convert.text_to_string(self._raw_data.labels[index])
        for index in indices:
            yield get_label(index).strip(), index

    def _find_selection_indices(self, selection):
        if selection == select.all:
            return range(len(self._raw_data.labels))
        else:
            selection_parts = _split_selection_in_parts(selection)
            return [self._find_selection_index(part) for part in selection_parts]

    def _find_selection_index(self, selection):
        for index, label in enumerate(self._raw_data.labels):
            label = convert.text_to_string(label).strip()
            if selection in label:
                return index
        raise exception.IncorrectUsage(
            f"{selection} was not found in the list of energies. "
            "Please make sure the spelling is correct."
        )

    def _create_yaxes(self, selection):
        return _YAxes(self._parse_user_selection(selection))

    def _make_series(self, yaxes, selection):
        steps = np.arange(len(self._raw_data.values))[self._slice] + 1
        return [
            graph.Series(
                x=steps,
                y=self._raw_data.values[self._slice, index],
                name=label[:14].strip(),
                y2=yaxes.y2(label),
            )
            for label, index in self._parse_user_selection(selection)
        ]


class _YAxes:
    def __init__(self, selections):
        selections = set(self._is_temperature(s) for s, _ in selections)
        use_energy = False in selections
        self.use_both = len(selections) == 2
        self.ylabel = "Energy (eV)" if use_energy else "Temperature (K)"
        self.y2label = "Temperature (K)" if self.use_both else None

    def y2(self, label):
        return self.use_both and self._is_temperature(label)

    def _is_temperature(self, label):
        return "temperature" in label


def _split_selection_in_parts(selection):
    error_message = "Energy selection must be a string."
    check.raise_error_if_not_string(selection, error_message)
    return (part.strip() for part in selection.split(","))


def _unpack_if_only_one_element(tuple_):
    if len(tuple_) == 1:
        return tuple_[0]
    else:
        return tuple_
