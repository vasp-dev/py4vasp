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
    width: np.ndarray = None
    y2: bool = False

    def to_plotly(self):
        return go.Scatter(x=self.x, y=self.y, name=self.name)


@dataclass
class Graph:
    series: Series or Sequence[Series]
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

    def _figure_with_one_or_two_y_axes(self):
        if any(series.y2 for series in np.atleast_1d(self.series)):
            return make_subplots(specs=[[{"secondary_y": True}]])
        else:
            return go.Figure()

    def _trace_generator(self):
        colors = itertools.cycle(_vasp_colors)
        for series in np.atleast_1d(self.series):
            factory = _PlotlyTraceFactory(color=next(colors))
            for y in np.atleast_2d(series.y.T):
                yield factory.make_trace(replace(series, y=y))
                factory.first_trace = False


@dataclass
class _PlotlyTraceFactory:
    color: str
    first_trace: bool = True

    def make_trace(self, series):
        if series.width is None:
            options = self._options_line(series)
        else:
            options = self._options_area(series)
        return go.Scatter(**options)

    def _options_line(self, series):
        return {
            **self._common_options(series),
            "x": series.x,
            "y": series.y,
            "line": {"color": self.color},
        }

    def _options_area(self, series):
        upper = series.y + series.width
        lower = series.y - series.width
        return {
            **self._common_options(series),
            "x": np.concatenate((series.x, series.x[::-1])),
            "y": np.concatenate((lower, upper[::-1])),
            "mode": "none",
            "fill": "toself",
            "fillcolor": self.color,
            "opacity": 0.5,
        }

    def _common_options(self, series):
        return {
            "name": series.name,
            "legendgroup": series.name,
            "showlegend": self.first_trace,
            "yaxis": "y2" if series.y2 else "y",
        }
