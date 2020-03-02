import plotly.graph_objects as go
from py4vasp.data import _util


class Convergence:
    def __init__(self, raw_conv):
        self._raw = raw_conv

    @classmethod
    def from_file(cls, file=None):
        return _util.from_file(cls, file, "convergence")

    def read(self, *args):
        return self.to_dict(*args)

    def plot(self, *args):
        return self.to_plotly(*args)

    def to_dict(self, selection=None):
        if selection is None:
            selection = "TOTEN"
        for i, label in enumerate(self._raw.labels):
            label = str(label, "utf-8").strip()
            if selection in label:
                return {label: self._raw.energies[:, i]}

    def to_plotly(self, selection=None):
        label, data = self.read(selection).popitem()
        label = "Temperature (K)" if "TEIN" in label else "Energy (eV)"
        data = go.Scatter(y=data)
        default = {
            "xaxis": {"title": {"text": "Step"}},
            "yaxis": {"title": {"text": label}},
        }
        return go.Figure(data=data, layout=default)
