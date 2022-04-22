from dataclasses import dataclass, replace
import itertools
import numpy as np
from typing import Sequence, NamedTuple
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
    color: str = None

    def _generate_traces(self):
        first_trace = True
        for y in np.atleast_2d(self.y.T):
            yield self._make_trace(y, first_trace)
            first_trace = False

    def _make_trace(self, y, first_trace):
        if self.width is None:
            options = self._options_line(y, first_trace)
        else:
            options = self._options_area(y, first_trace)
        return go.Scatter(**options)

    def _options_line(self, y, first_trace):
        return {
            **self._common_options(first_trace),
            "x": self.x,
            "y": y,
            "line": {"color": self.color},
        }

    def _options_area(self, y, first_trace):
        upper = y + self.width
        lower = y - self.width
        return {
            **self._common_options(first_trace),
            "x": np.concatenate((self.x, self.x[::-1])),
            "y": np.concatenate((lower, upper[::-1])),
            "mode": "none",
            "fill": "toself",
            "fillcolor": self.color,
            "opacity": 0.5,
        }

    def _common_options(self, first_trace):
        return {
            "name": self.name,
            "legendgroup": self.name,
            "showlegend": first_trace,
            "yaxis": "y2" if self.y2 else "y",
        }


@dataclass
class Graph:
    series: Series or Sequence[Series]
    xlabel: str = None
    xticks: dict = None
    ylabel: str = None
    y2label: str = None
    title: str = None

    def to_plotly(self):
        figure = self._make_plotly_figure()
        for trace in self._generate_plotly_traces():
            figure.add_trace(trace)
        return figure

    def _ipython_display_(self):
        self.to_plotly()._ipython_display_()

    def _generate_plotly_traces(self):
        colors = itertools.cycle(_vasp_colors)
        for series in np.atleast_1d(self.series):
            if not series.color:
                series = replace(series, color=next(colors))
            yield from series._generate_traces()

    def _make_plotly_figure(self):
        figure = self._figure_with_one_or_two_y_axes()
        self._set_xaxis_options(figure)
        self._set_yaxis_options(figure)
        figure.layout.title.text = self.title
        return figure

    def _figure_with_one_or_two_y_axes(self):
        if any(series.y2 for series in np.atleast_1d(self.series)):
            return make_subplots(specs=[[{"secondary_y": True}]])
        else:
            return go.Figure()

    def _set_xaxis_options(self, figure):
        figure.layout.xaxis.title.text = self.xlabel
        if self.xticks:
            figure.layout.xaxis.tickmode = "array"
            figure.layout.xaxis.tickvals = tuple(self.xticks.keys())
            figure.layout.xaxis.ticktext = tuple(self.xticks.values())

    def _set_yaxis_options(self, figure):
        figure.layout.yaxis.title.text = self.ylabel
        if self.y2label:
            figure.layout.yaxis2.title.text = self.y2label
