from dataclasses import dataclass
import numpy as np
from typing import Sequence
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class Series:
    x: np.ndarray
    y: np.ndarray
    name: str
    y2: bool = False

    def to_plotly(self):
        return go.Scatter(x=self.x, y=self.y, name=self.name)


@dataclass
class Graph:
    data: Series or Sequence[Series]
    xlabel: str = None
    ylabel: str = None
    y2label: str = None

    def to_plotly(self):
        figure = self._figure_with_one_or_two_y_axes()
        figure.layout.xaxis.title.text = self.xlabel
        figure.layout.yaxis.title.text = self.ylabel
        if self.y2label:
            figure.layout.yaxis2.title.text = self.y2label
        for series in self._series_generator():
            trace = go.Scatter(x=series.x, y=series.y, name=series.name)
            figure.add_trace(trace, **self._figure_options(series))
        return figure

    def _series_generator(self):
        try:
            yield from self.data
        except TypeError:
            yield self.data

    def _figure_with_one_or_two_y_axes(self):
        if any(series.y2 for series in self._series_generator()):
            return make_subplots(specs=[[{"secondary_y": True}]])
        else:
            return go.Figure()

    def _figure_options(self, series):
        options = {}
        if series.y2:
            options["secondary_y"] = True
        return options
