import plotly.graph_objects as go
import functools
from py4vasp.data import _util
import py4vasp.exceptions as exception


@_util.add_wrappers
class Energy(_util.Data):
    """ The energy data for all steps of a relaxation or MD simulation.

    You can use this class to inspect how the ionic relaxation converges or
    during an MD simulation whether the total energy is conserved.

    Parameters
    ----------
    raw_energy : raw.Energy
        Dataclass containing the raw energy values for the ionic run and labels
        specifying which energies are stored.
    """

    def __init__(self, raw_energy):
        super().__init__(raw_energy)

    @classmethod
    @_util.add_doc(_util.from_file_doc("energies in an relaxation or MD simulation"))
    def from_file(cls, file=None):
        return _util.from_file(cls, file, "energy")

    def _repr_pretty_(self, p, cycle):
        text = "Energies at last step:"
        for label, value in zip(self._raw.labels, self._raw.values[-1]):
            label = f"{_util.decode_if_possible(label):22.22}"
            text += f"\n   {label}={value:17.6f}"
        p.text(text)

    def to_dict(self, selection=None):
        """ Read the energy data and store it in a dictionary.

        Parameters
        ----------
        selection : str or None
            String specifying the label of the energy to be plotted. A substring
            of the label is sufficient. If no energy is select this will default
            to the total energy.

        Returns
        -------
        dict
            Contains the exact label corresponding to the selection and the
            associated energy for every ionic step.
        """
        if selection is None:
            selection = "TOTEN"
        error_message = "Energy selection must be a string."
        _util.raise_error_if_not_string(selection, error_message)
        for i, label in enumerate(self._raw.labels):
            label = _util.decode_if_possible(label).strip()
            if selection in label:
                return {label: self._raw.values[:, i]}
        else:
            raise exception.IncorrectUsage(
                f"{selection} was not found in the list of energies. "
                "Please make sure the spelling is correct."
            )

    def to_plotly(self, selection=None):
        """ Read the energy data and generate a plotly figure.

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
        label, data = self.read(selection).popitem()
        label = "Temperature (K)" if "TEIN" in label else "Energy (eV)"
        data = go.Scatter(y=data)
        default = {
            "xaxis": {"title": {"text": "Step"}},
            "yaxis": {"title": {"text": label}},
        }
        return go.Figure(data=data, layout=default)
