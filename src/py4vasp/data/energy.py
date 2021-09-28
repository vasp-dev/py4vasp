import plotly.graph_objects as go
from plotly.subplots import make_subplots
import functools
import numpy as np
from py4vasp.data._base import RefinementDescriptor
import py4vasp.data._export as _export
import py4vasp.data._trajectory as _trajectory
import py4vasp.exceptions as exception
import py4vasp._util.documentation as _documentation
import py4vasp._util.convert as _convert
import py4vasp._util.sanity_check as _check


_energy_docs = f"""
The energy data for one or several steps of a relaxation or MD simulation.

You can use this class to inspect how the ionic relaxation converges or
during an MD simulation whether the total energy is conserved.

Parameters
----------
raw_energy : RawEnergy
    Dataclass containing the raw energy values for the ionic run and labels
    specifying which energies are stored.

{_trajectory.trajectory_examples("energy")}
""".strip()


@_documentation.add(_energy_docs)
class Energy(_trajectory.DataTrajectory, _export.Image):
    read = RefinementDescriptor("_to_dict")
    to_dict = RefinementDescriptor("_to_dict")
    plot = RefinementDescriptor("_to_plotly")
    to_plotly = RefinementDescriptor("_to_plotly")
    to_numpy = RefinementDescriptor("_to_numpy")
    __str__ = RefinementDescriptor("_to_string")

    def _to_string(self):
        text = "Energies at last step:"
        for label, value in zip(self._raw_data.labels, self._raw_data.values[-1]):
            label = f"{_convert.text_to_string(label):22.22}"
            text += f"\n   {label}={value:17.6f}"
        return text

    def _to_dict(self, selection=None):
        """Read the energy data and store it in a dictionary.

        Parameters
        ----------
        selection : str or None
            String specifying the labels of the energy to be read. A substring
            of the label is sufficient. If no energy is select this will default
            to the total energy. Separate distinct labels by commas.

        Returns
        -------
        dict
            Contains the exact labels corresponding to the selection and the
            associated energies for every selected ionic step.
        """
        return {
            label: self._raw_data.values[self._steps, index]
            for label, index in self._parse_selection(selection)
        }

    def _to_plotly(self, selection=None):
        """Read the energy data and generate a plotly figure.

        Parameters
        ----------
        selection : str or None
            String specifying the labels of the energy to be plotted. A substring
            of the label is sufficient. If no energy is select this will default
            to the total energy. Separate distinct labels by commas.

        Returns
        -------
        plotly.graph_objects.Figure
            plotly figure containing the selected energies for every selected ionic step.
        """
        figure, use_secondary = self._create_figure_with_yaxes(selection)
        figure.layout.xaxis.title.text = "Step"
        steps = np.arange(len(self._raw_data.values))[self._slice] + 1
        for label, index in self._parse_selection(selection):
            short_label = label[:14].strip()
            data = self._raw_data.values[self._slice, index]
            options = {"secondary_y": True} if use_secondary(label) else {}
            figure.add_trace(go.Scatter(x=steps, y=data, name=short_label), **options)
        return figure

    def _to_numpy(self, selection=None):
        """Read the energy of the selected steps.

        Parameters
        ----------
        selection : str or None
            String specifying the labels of the energy to be read. A substring
            of the label is sufficient. If no energy is select this will default
            to the total energy. Separate distinct labels by commas.

        Returns
        -------
        float or np.ndarray or tuple
            Contains energies associated with the selection for the selected ionic step(s).
            When only a single step is inquired, result is a float otherwise an array.
            If you select multiple quantities a tuple of them is returned.
        """
        result = tuple(
            self._raw_data.values[self._steps, index]
            for _, index in self._parse_selection(selection)
        )
        return _unpack_if_only_one_element(result)

    def _parse_selection(self, selection):
        # TODO check if SelectionTree can be used instead
        indices = self._find_selection_indices(selection)
        get_label = lambda index: _convert.text_to_string(self._raw_data.labels[index])
        for index in indices:
            yield get_label(index).strip(), index

    def _find_selection_indices(self, selection):
        selection_parts = _actual_or_default_selection(selection)
        return [self._find_selection_index(part) for part in selection_parts]

    def _find_selection_index(self, selection):
        for index, label in enumerate(self._raw_data.labels):
            label = _convert.text_to_string(label).strip()
            if selection in label:
                return index
        raise exception.IncorrectUsage(
            f"{selection} was not found in the list of energies. "
            "Please make sure the spelling is correct."
        )

    def _create_figure_with_yaxes(self, selection):
        use_temperature, use_energy = self._select_yaxes(selection)
        if use_temperature and use_energy:
            figure = make_subplots(specs=[[{"secondary_y": True}]])
            _set_yaxis_text(figure.layout.yaxis, use_temperature=False)
            _set_yaxis_text(figure.layout.yaxis2, use_temperature=True)
            use_secondary = lambda label: _is_temperature(label)
        else:
            figure = go.Figure()
            _set_yaxis_text(figure.layout.yaxis, use_temperature)
            use_secondary = lambda label: False
        return figure, use_secondary

    def _select_yaxes(self, selection):
        use_temperature = False
        use_energy = False
        for label, _ in self._parse_selection(selection):
            if _is_temperature(label):
                use_temperature = True
            else:
                use_energy = True
        return use_temperature, use_energy


def _is_temperature(label):
    return "temperature" in label


def _set_yaxis_text(yaxis, use_temperature):
    if use_temperature:
        yaxis.title.text = "Temperature (K)"
    else:
        yaxis.title.text = "Energy (eV)"


def _actual_or_default_selection(selection):
    if selection is not None:
        error_message = "Energy selection must be a string."
        _check.raise_error_if_not_string(selection, error_message)
        return (part.strip() for part in selection.split(","))
    else:
        return ("TOTEN",)


def _unpack_if_only_one_element(tuple_):
    if len(tuple_) == 1:
        return tuple_[0]
    else:
        return tuple_
