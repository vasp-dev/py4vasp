# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from dataclasses import dataclass, fields

import numpy as np
import plotly.graph_objects as go

from py4vasp import exception


@dataclass
class Series:
    """Represents a single series in a graph.

    Typically this corresponds to a single line of x-y data with an optional name used
    in the legend of the figure. The look of the series is modified by some of the other
    optional arguments.
    """

    x: np.ndarray
    "The x coordinates of the series."
    y: np.ndarray
    """The y coordinates of the series. If the data is 2-dimensional multiple lines are
    generated with a common entry in the legend."""
    name: str = None
    "A label for the series used in the legend."
    width: np.ndarray = None
    "When a width is set, the series will be visualized as an area instead of a line."
    y2: bool = False
    "Use a secondary y axis to show this series."
    subplot: int = None
    "Split series into different axes"
    color: str = None
    "The color used for this series."
    marker: str = None
    "Which marker is used for the series, None defaults to line mode."
    _frozen = False

    def __post_init__(self):
        if len(self.x) != np.array(self.y).shape[-1]:
            message = "The length of the two plotted components is inconsistent."
            raise exception.IncorrectUsage(message)
        if self.width is not None and len(self.x) != self.width.shape[-1]:
            message = "The length of width and plot is inconsistent."
            raise exception.IncorrectUsage(message)
        self._frozen = True

    def __setattr__(self, key, value):
        # prevent adding new attributes to avoid typos, in Python 3.10 this could be
        # handled by setting slots=True when creating the dataclass
        assert not self._frozen or hasattr(self, key)
        super().__setattr__(key, value)

    def _generate_traces(self):
        first_trace = True
        for item in enumerate(np.atleast_2d(np.array(self.y))):
            yield self._make_trace(*item, first_trace), {"row": self.subplot}
            first_trace = False

    def _make_trace(self, index, y, first_trace):
        width = self._get_width(index)
        if self._is_line():
            options = self._options_line(y, first_trace)
        elif self._is_area():
            options = self._options_area(y, width, first_trace)
        else:
            options = self._options_points(y, width, first_trace)
        return go.Scatter(**options)

    def _get_width(self, index):
        if self.width is None:
            return None
        elif self.width.ndim == 1:
            return self.width
        else:
            return self.width[index]

    def _is_line(self):
        return (self.width is None) and (self.marker is None)

    def _is_area(self):
        return (self.width is not None) and (self.marker is None)

    def _options_line(self, y, first_trace):
        return {
            **self._common_options(first_trace),
            "x": self.x,
            "y": y,
            "line": {"color": self.color},
        }

    def _options_area(self, y, width, first_trace):
        upper = y + width
        lower = y - width
        return {
            **self._common_options(first_trace),
            "x": np.concatenate((self.x, self.x[::-1])),
            "y": np.concatenate((lower, upper[::-1])),
            "mode": "none",
            "fill": "toself",
            "fillcolor": self.color,
            "opacity": 0.5,
        }

    def _options_points(self, y, width, first_trace):
        return {
            **self._common_options(first_trace),
            "x": self.x,
            "y": y,
            "mode": "markers",
            "marker": {"size": width, "sizemode": "area", "color": self.color},
        }

    def _common_options(self, first_trace):
        return {
            "name": self.name,
            "legendgroup": self.name,
            "showlegend": first_trace,
            "yaxis": "y2" if self.y2 else "y",
        }


Series._fields = tuple(field.name for field in fields(Series))
