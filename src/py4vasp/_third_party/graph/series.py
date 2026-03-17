# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

from py4vasp import exception
from py4vasp._third_party.graph import trace
from py4vasp._util import import_

go = import_.optional("plotly.graph_objects")


@dataclass
class Series(trace.Trace):
    """Represents a single series in a graph.

    A Series object encapsulates x-y data for plotting, with support for multiple plot
    types (lines, areas, scatter points) and customization options.

    Examples
    --------
    Create a simple line plot:
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x)
    >>> series = Series(x=x, y=y, label="Sine wave")

    Create scatter plot with custom color and marker:
    >>> series = Series(x=[1, 2, 3], y=[4, 5, 6], label="Data", color="blue",
    ...     marker="circle")

    Add weighted points (size based on weight):
    >>> series = Series(x=[1, 2, 3], y=[4, 5, 6], weight=[10, 20, 30],
    ...     weight_mode="size")

    Add hover annotations:
    >>> series = Series(x=[1, 2, 3], y=[4, 5, 6], annotations={
    ...     "Temperature": [20, 25, 30], "Unit": "°C"})

    Plot multiple lines (2D y data):
    >>> y_multi = np.array([[1, 2, 3], [4, 5, 6]])
    >>> series = Series(x=[0, 1, 2], y=y_multi, label="Multi-line")

    Create area plot with uncertainty:
    >>> series = Series(x=[1, 2, 3], y=[10, 20, 15], weight=[2, 3, 2.5],
    ...     color="rgba(255,0,0,0.3)")
    """

    x: np.ndarray
    """The x coordinates of the series."""
    y: np.ndarray
    """The y coordinates of the series. If the data is 2-dimensional, multiple lines are
    generated with a common entry in the legend."""
    label: Optional[str] = None
    """A label for the series used in the legend."""
    weight: Optional[np.ndarray] = None
    """Optional weights that modify the plot according to weight_mode. Can be used to
    show uncertainty bands, scale marker sizes, or color-code points."""
    weight_mode: str = "size"
    """Controls how weights affect the visualization:
    - 'size': Adjusts the size of markers based on weight values
    - 'color': Adjusts the color of markers based on weight values (uses coloraxis)
    """
    annotations: Optional[dict] = None
    """Optional metadata dictionary for hover text. Keys are labels and values are either
    scalars (same for all points) or arrays matching the length of x coordinates.
    Displayed on hover as key-value pairs."""
    y2: bool = False
    """If True, plot this series on a secondary y-axis (right side)."""
    subplot: Optional[int] = None
    """Subplot index for this series. If specified, the series is plotted in a separate
    subplot rather than overlaid with others."""
    color: Optional[str] = None
    """The color used for this series. Accepts any valid CSS color string, including
    rgba() for transparency in area plots."""
    marker: Optional[str] = None
    """Marker style for scatter plots. If None, displays as a line plot. Common values
    include 'circle', 'square', 'diamond', etc."""
    _frozen = False

    def __post_init__(self):
        self.x = np.asarray(self.x)
        self.y = np.asarray(self.y)
        if len(self.x) != self.y.shape[-1]:
            message = "The length of the two plotted components is inconsistent."
            raise exception.IncorrectUsage(message)
        if self.weight is not None:
            self.weight = np.asarray(self.weight)
            if len(self.x) != self.weight.shape[-1]:
                message = "The length of weight and plot is inconsistent."
                raise exception.IncorrectUsage(message)
        self._raise_error_if_annotations_length_incorrect()
        self._frozen = True

    def _raise_error_if_annotations_length_incorrect(self):
        if self.annotations is None:
            return
        for key, value in self.annotations.items():
            if np.ndim(value) == 0:
                continue
            if len(value) != len(self.x):
                message = f"The length of annotation '{key}' must be 1 or match the length of x."
                raise exception.IncorrectUsage(message)

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
            "text": self._convert_annotations(),
            "legendgroup": self.label,
            "showlegend": first_trace,
            "yaxis": "y2" if self.y2 else "y",
        }

    def _convert_annotations(self):
        if self.annotations is None:
            return None
        return [self._convert_annotation(index_) for index_ in range(len(self.x))]

    def _convert_annotation(self, index_):
        return "<br>".join(
            f"{key}: {self._get_element(value, index_)}"
            for key, value in self.annotations.items()
        )

    def _get_element(self, array_or_scalar, index_):
        if np.ndim(array_or_scalar) == 0:
            return array_or_scalar
        return array_or_scalar[index_]

    def _generate_shapes(self):
        return ()


Series._fields = tuple(field.name for field in fields(Series))
