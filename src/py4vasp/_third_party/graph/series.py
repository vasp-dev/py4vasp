# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from dataclasses import dataclass, fields

import numpy as np

from py4vasp import exception
from py4vasp._third_party.graph import trace
from py4vasp._util import import_

go = import_.optional("plotly.graph_objects")


@dataclass
class Series(trace.Trace):
    """Represents a single series in a graph.

    Typically this corresponds to a single line of x-y data with an optional label used
    in the legend of the figure. The look of the series is modified by some of the other
    optional arguments.
    """

    x: np.ndarray
    "The x coordinates of the series."
    y: np.ndarray
    """The y coordinates of the series. If the data is 2-dimensional multiple lines are
    generated with a common entry in the legend."""
    label: str = None
    "A label for the series used in the legend."
    weight: np.ndarray = None
    "When a weight is set, the series will modify the plot according to weight_mode."
    weight_mode: str = "size"
    """If weight_mode is 'size', the size of the plot is adjusted according to the weight.
    If weight_mode is 'color', the color of the plot is adjusted according to the weight."""
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
        self.x = np.asarray(self.x)
        self.y = np.asarray(self.y)
        if len(self.x) != self.y.shape[-1]:
            message = "The length of the two plotted components is inconsistent."
            raise exception.IncorrectUsage(message)
        if self.weight is not None and len(self.x) != self.weight.shape[-1]:
            message = "The length of weight and plot is inconsistent."
            raise exception.IncorrectUsage(message)
        self._frozen = True

    def __setattr__(self, key, value):
        # prevent adding new attributes to avoid typos, in Python 3.10 this could be
        # handled by setting slots=True when creating the dataclass
        assert not self._frozen or hasattr(self, key)
        super().__setattr__(key, value)

    def __eq__(self, other):
        if not isinstance(other, Series):
            return NotImplemented
        return all(
            np.array_equal(getattr(self, field.name), getattr(other, field.name))
            for field in fields(self)
        )

    def to_plotly(self):
        first_trace = True
        for item in enumerate(np.atleast_2d(np.array(self.y))):
            yield self._make_trace(*item, first_trace), {"row": self.subplot}
            first_trace = False

    def _make_trace(self, index, y, first_trace):
        weight = self._get_weight(index)
        if self._is_line():
            specific_options = self._options_line(y)
        elif self._is_area():
            specific_options = self._options_area(y, weight)
        elif self.weight_mode == "size":
            specific_options = self._options_scaled_points(y, weight)
        else:
            specific_options = self._options_colored_points(y, weight)
        return go.Scatter(**self._common_options(first_trace), **specific_options)

    def _get_weight(self, index):
        if self.weight is None:
            return None
        elif self.weight.ndim == 1:
            return self.weight
        else:
            return self.weight[index]

    def _is_line(self):
        return (self.weight is None) and (self.marker is None)

    def _is_area(self):
        return (self.weight is not None) and (self.marker is None)

    def _options_line(self, y):
        return {
            "x": self.x,
            "y": y,
            "line": {"color": self.color},
        }

    def _options_area(self, y, weight):
        upper = y + weight
        lower = y - weight
        return {
            "x": np.concatenate((self.x, self.x[::-1])),
            "y": np.concatenate((lower, upper[::-1])),
            "mode": "none",
            "fill": "toself",
            "fillcolor": self.color,
            "opacity": 0.5,
        }

    def _options_scaled_points(self, y, weight):
        return {
            "x": self.x,
            "y": y,
            "mode": "markers",
            "marker": {"size": weight, "sizemode": "area", "color": self.color},
        }

    def _options_colored_points(self, y, weight):
        return {
            "x": self.x,
            "y": y,
            "mode": "markers",
            "marker": {"color": weight, "coloraxis": "coloraxis"},
        }

    def _common_options(self, first_trace):
        return {
            "name": self.label,
            "legendgroup": self.label,
            "showlegend": first_trace,
            "yaxis": "y2" if self.y2 else "y",
        }

    def _generate_shapes(self):
        return ()


Series._fields = tuple(field.name for field in fields(Series))
