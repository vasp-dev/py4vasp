import plotly.graph_objects as go
from contextlib import contextmanager
import py4vasp.raw as raw


class Convergence:
    def __init__(self, raw_conv):
        self._raw = raw_conv

    @classmethod
    @contextmanager
    def from_file(cls, file=None):
        if file is None or isinstance(file, str):
            with raw.File(file) as local_file:
                yield cls(local_file.convergence())
        else:
            yield cls(file.convergence())

    def read(self, selection=None):
        if selection is None:
            selection = "TOTEN"
        for i, label in enumerate(self._raw.labels):
            label = str(label, "utf-8").strip()
            if selection in label:
                return label, self._raw.energies[:, i]

    def plot(self, selection=None):
        label, data = self.read(selection)
        label = "Temperature (K)" if "TEIN" in label else "Energy (eV)"
        data = go.Scatter(y=data)
        default = {
            "xaxis": {"title": {"text": "Step"}},
            "yaxis": {"title": {"text": label}},
        }
        return go.Figure(data=data, layout=default)
