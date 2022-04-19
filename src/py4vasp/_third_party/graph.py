from dataclasses import dataclass
import numpy as np
from typing import Sequence
import plotly.graph_objects as go


@dataclass
class Series:
    x: np.ndarray
    y: np.ndarray
    name: str

    def to_plotly(self):
        return go.Scatter(x=self.x, y=self.y, name=self.name)


@dataclass
class Graph:
    data: Series or Sequence[Series]
    xlabel: str = None
    ylabel: str = None

    def to_plotly(self):
        data = [series.to_plotly() for series in self.series_generator()]
        layout = {
            "xaxis": {"title": {"text": self.xlabel}},
            "yaxis": {"title": {"text": self.ylabel}},
        }
        return go.Figure(data=data, layout=layout)

    def series_generator(self):
        try:
            yield from self.data
        except TypeError:
            yield self.data
