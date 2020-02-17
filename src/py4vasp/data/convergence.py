import plotly.graph_objects as go


class Convergence:
    def __init__(self, raw_conv):
        self._conv = raw_conv

    def read(self, selection=None):
        if selection is None:
            selection = "TOTEN"
        for i, label in enumerate(self._conv.labels):
            label = str(label, "utf-8").strip()
            if selection in label:
                return label, self._conv.energies[:, i]

    def plot(self, selection=None):
        label, data = self.read(selection)
        label = "Temperature (K)" if "TEIN" in label else "Energy (eV)"
        data = go.Scatter(y=data)
        default = {
            "xaxis": {"title": {"text": "Step"}},
            "yaxis": {"title": {"text": label}},
        }
        return go.Figure(data=data, layout=default)
