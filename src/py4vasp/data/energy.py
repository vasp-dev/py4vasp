import plotly.graph_objects as go
import functools
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
        String specifying the label of the energy to be read. A substring
        of the label is sufficient. If no energy is select this will default
        to the total energy.

    Returns
    -------
    dict
        Contains the exact label corresponding to the selection and the
        associated energy for every ionic step.
    """
    index = _find_selection_index(raw_energy, selection)
    label = _util.decode_if_possible(raw_energy.labels[index]).strip()
    return {label: raw_energy.values[:, index]}


def _to_plotly(raw_energy, selection=None):
    """Read the energy data and generate a plotly figure.

    Parameters
    ----------
    selection : str or None
        String specifying the label of the energy to be plotted. A substring
        of the label is sufficient. If no energy is select this will default
        to the total energy.

    Returns
    -------
    plotly.graph_objects.Figure
        plotly figure containing the selected energy for every ionic step.
    """
    label, data = _to_dict(raw_energy, selection).popitem()
    label = "Temperature (K)" if "TEIN" in label else "Energy (eV)"
    data = go.Scatter(y=data)
    default = {
        "xaxis": {"title": {"text": "Step"}},
        "yaxis": {"title": {"text": label}},
    }
    return go.Figure(data=data, layout=default)


def _final(raw_energy, selection=None):
    """Read the energy of the final iteration.

    Parameters
    ----------
    selection : str or None
        String specifying the label of the energy to be read. A substring
        of the label is sufficient. If no energy is select this will default
        to the total energy.

    Returns
    -------
    float
        Contains energy associated with the selection for the final ionic step.
    """
    index = _find_selection_index(raw_energy, selection)
    return raw_energy.values[-1, index]


def _find_selection_index(raw_energy, selection):
    selection = _actual_or_default_selection(selection)
    for index, label in enumerate(raw_energy.labels):
        label = _util.decode_if_possible(label).strip()
        if selection in label:
            return index
    raise exception.IncorrectUsage(
        f"{selection} was not found in the list of energies. "
        "Please make sure the spelling is correct."
    )


def _actual_or_default_selection(selection):
    if selection is not None:
        error_message = "Energy selection must be a string."
        _util.raise_error_if_not_string(selection, error_message)
        return selection
    else:
        return "TOTEN"
