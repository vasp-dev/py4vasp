import plotly.graph_objects as go
from plotly.subplots import make_subplots
import functools
import numpy as np
from py4vasp.data import _util
from py4vasp.data._base import DataBase, RefinementDescriptor
import py4vasp.exceptions as exception


class Energy(DataBase):
    """The energy data for all steps of a relaxation or MD simulation.

    You can use this class to inspect how the ionic relaxation converges or
    during an MD simulation whether the total energy is conserved.

    Parameters
    ----------
    raw_energy : RawEnergy
        Dataclass containing the raw energy values for the ionic run and labels
        specifying which energies are stored.
    """

    read = RefinementDescriptor("_to_dict")
    to_dict = RefinementDescriptor("_to_dict")
    plot = RefinementDescriptor("_to_plotly")
    to_plotly = RefinementDescriptor("_to_plotly")
    final = RefinementDescriptor("_final")
    __str__ = RefinementDescriptor("_to_string")


def _to_string(raw_energy):
    text = "Energies at last step:"
    for label, value in zip(raw_energy.labels, raw_energy.values[-1]):
        label = f"{_util.decode_if_possible(label):22.22}"
        text += f"\n   {label}={value:17.6f}"
    return text


def _to_dict(raw_energy, selection=None):
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
        associated energies for every ionic step.
    """
    indices = _find_selection_indices(raw_energy, selection)
    get_label = lambda index: _util.decode_if_possible(raw_energy.labels[index]).strip()
    return {get_label(index): raw_energy.values[:, index] for index in indices}


def _to_plotly(raw_energy, selection=None):
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
        plotly figure containing the selected energies for every ionic step.
    """
    dict_ = _to_dict(raw_energy, selection)
    figure = _create_yaxes(dict_)
    figure.layout.xaxis.title.text = "Step"
    use_secondary = lambda label: _is_temperature(label) and len(dict_) > 1
    for label, data in dict_.items():
        steps = np.arange(len(data)) + 1
        short_label = label[:14].strip()
        options = {"secondary_y": True} if use_secondary(label) else {}
        figure.add_trace(go.Scatter(x=steps, y=data, name=short_label), **options)
    return figure


def _final(raw_energy, selection=None):
    """Read the energy of the final iteration.

    Parameters
    ----------
    selection : str or None
        String specifying the labels of the energy to be read. A substring
        of the label is sufficient. If no energy is select this will default
        to the total energy. Separate distinct labels by commas.

    Returns
    -------
    float or np.ndarray
        Contains energies associated with the selection for the final ionic step.
        When only a single quantity is inquired, result is a float otherwise an array.
    """
    indices = np.array(_find_selection_indices(raw_energy, selection))
    if len(indices) == 1:
        return raw_energy.values[-1, indices[0]]
    else:
        return raw_energy.values[-1, indices]


def _find_selection_indices(raw_energy, selection):
    selection_parts = _actual_or_default_selection(selection)
    return [_find_selection_index(raw_energy, part) for part in selection_parts]


def _actual_or_default_selection(selection):
    if selection is not None:
        error_message = "Energy selection must be a string."
        _util.raise_error_if_not_string(selection, error_message)
        return (part.strip() for part in selection.split(","))
    else:
        return ("TOTEN",)


def _find_selection_index(raw_energy, selection):
    for index, label in enumerate(raw_energy.labels):
        label = _util.decode_if_possible(label).strip()
        if selection in label:
            return index
    raise exception.IncorrectUsage(
        f"{selection} was not found in the list of energies. "
        "Please make sure the spelling is correct."
    )


def _create_yaxes(dict_):
    secondary_axis = any(_is_temperature(label) for label in dict_) and len(dict_) > 1
    if secondary_axis:
        figure = make_subplots(specs=[[{"secondary_y": True}]])
        _set_yaxis_text(figure.layout.yaxis, is_temperature=False)
        _set_yaxis_text(figure.layout.yaxis2, is_temperature=True)
    else:
        figure = go.Figure()
        _set_yaxis_text(figure.layout.yaxis, _is_temperature(list(dict_.keys())[0]))
    return figure


def _is_temperature(label):
    return "temperature" in label


def _set_yaxis_text(yaxis, is_temperature):
    if is_temperature:
        yaxis.title.text = "Temperature (K)"
    else:
        yaxis.title.text = "Energy (eV)"
