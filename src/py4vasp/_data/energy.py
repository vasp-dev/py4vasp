# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import exception
from py4vasp._data import base, slice_
from py4vasp._third_party import graph
from py4vasp._util import check, convert, documentation, index, select


def _selection_string(default):
    return f"""selection : str or None
    String specifying the labels of the energy to be read. A substring
    of the label is sufficient. If no energy is select this will default
    to selecting {default}. Separate distinct labels by commas. For a
    complete list of all possible selections, please use
    >>> calc.energy.labels()
"""


_SELECTIONS = {
    "ion-electron   TOTEN   ": ["ion_electron", "TOTEN"],
    "kinetic energy EKIN    ": ["kinetic_energy", "EKIN"],
    "kin. lattice   EKIN_LAT": ["kinetic_lattice", "EKIN_LAT"],
    "temperature    TEIN    ": ["temperature", "TEIN"],
    "nose potential ES      ": ["nose_potential", "ES"],
    "nose kinetic   EPS     ": ["nose_kinetic", "EPS"],
    "total energy   ETOTAL  ": ["total_energy", "ETOTAL"],
}


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
            label = f"{convert.text_to_string(label):23.23}"
            text += f"\n   {label}={value:17.6f}"
        return text

    def _step_string(self):
        if self._is_slice:
            range_ = range(len(self._raw_data.values))[self._slice]
            start = range_.start + 1  # convert to Fortran index
            stop = range_.stop
            return f"step {stop} of range {start}:{stop}"
        elif self._steps == -1:
            return "final step"
        else:
            return f"step {self._steps + 1}"

    @base.data_access
    @documentation.format(
        selection=_selection_string("all energies"),
        examples=slice_.examples("energy", "to_dict"),
    )
    def to_dict(self, selection=None):
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
        if selection is None:
            return self._default_dict()
        return dict(self._read_data(selection, self._steps))

    def _default_dict(self):
        return {
            convert.text_to_string(label).strip(): value[self._steps]
            for label, value in zip(self._raw_data.labels, self._raw_data.values.T)
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
        result = tuple(values for _, values in self._read_data(selection, self._steps))
        return np.array(_unpack_if_only_one_element(result))

    @base.data_access
    def selections(self):
        return tuple(self._init_selection_dict().keys())

    def _read_data(self, selection, steps_or_slice):
        tree = select.Tree.from_selection(selection)
        maps = {1: self._init_selection_dict()}
        selector = index.Selector(maps, self._raw_data.values)
        for selection in tree.selections():
            yield selector.label(selection), selector[selection][steps_or_slice]

    def _init_selection_dict(self):
        return {
            selection: index
            for index, label in enumerate(self._raw_data.labels)
            for selection in _SELECTIONS.get(convert.text_to_string(label), ())
        }

    def _make_series(self, yaxes, selection):
        steps = np.arange(len(self._raw_data.values))[self._slice] + 1
        return [
            graph.Series(
                x=steps,
                y=values,
                name=label,
                y2=yaxes.use_y2(label),
            )
            for label, values in self._read_data(selection, self._slice)
        ]

    def _create_yaxes(self, selection):
        tree = select.Tree.from_selection(selection)
        return _YAxes(tree)


class _YAxes:
    def __init__(self, tree):
        uses = set(self._is_temperature(selection) for selection in tree.selections())
        use_energy = False in uses
        self.use_both = len(uses) == 2
        self.ylabel = "Energy (eV)" if use_energy else "Temperature (K)"
        self.y2label = "Temperature (K)" if self.use_both else None

    def _is_temperature(self, selection):
        choices = _SELECTIONS["temperature    TEIN    "]
        return any(select.contains(selection, choice) for choice in choices)

    def use_y2(self, label):
        choices = _SELECTIONS["temperature    TEIN    "]
        return self.use_both and label in choices


def _unpack_if_only_one_element(tuple_):
    if len(tuple_) == 1:
        return tuple_[0]
    else:
        return tuple_
