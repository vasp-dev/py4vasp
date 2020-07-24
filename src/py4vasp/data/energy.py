import plotly.graph_objects as go
import functools
from py4vasp.data import _util


class Energy:
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
        self._raw = raw_energy

    @classmethod
    @_util.add_doc(_util.from_file_doc("energies in an relaxation or MD simulation"))
    def from_file(cls, file=None):
        return _util.from_file(cls, file, "energy")

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
        for i, label in enumerate(self._raw.labels):
            label = str(label, "utf-8").strip()
            if selection in label:
                return {label: self._raw.values[:, i]}

    @functools.wraps(to_dict)
    def read(self, *args):
        return self.to_dict(*args)

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

    @functools.wraps(to_plotly)
    def plot(self, *args):
        return self.to_plotly(*args)
