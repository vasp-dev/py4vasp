from dataclasses import dataclass, replace
import itertools
import numpy as np
from typing import Sequence
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_vasp_colors = ["#4C265F", "#2FB5AB", "#2C68FC", "#A82C35", "#808080"]


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
        for trace in self._trace_generator():
            figure.add_trace(trace)
        return figure

    def _trace_generator(self):
        colors = itertools.cycle(_vasp_colors)
        for series, color in zip(np.atleast_1d(self.data), colors):
            for i, y in enumerate(np.atleast_2d(series.y.T)):
                series = replace(series, y=y)
                trace = self._convert_series_to_trace(series)
                trace.showlegend = i == 0
                trace.line.color = color
                yield trace

    def _convert_series_to_trace(self, series):
        yaxis = "y2" if series.y2 else "y"
        return go.Scatter(
            x=series.x,
            y=series.y,
            yaxis=yaxis,
            name=series.name,
            legendgroup=series.name,
        )

    def _figure_with_one_or_two_y_axes(self):
        if any(series.y2 for series in np.atleast_1d(self.data)):
            return make_subplots(specs=[[{"secondary_y": True}]])
        else:
            return go.Figure()

    def _figure_options(self, series):
        options = {}
        if series.y2:
            options["secondary_y"] = True
        return options
