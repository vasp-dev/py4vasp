# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from collections.abc import Sequence
from dataclasses import dataclass, replace, fields
import itertools
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import py4vasp.exceptions as exception

_vasp_colors = ("#4C265F", "#2FB5AB", "#2C68FC", "#A82C35", "#808080")
pio.templates["vasp"] = go.layout.Template(layout={"colorway": _vasp_colors})
pio.templates.default = "ggplot2+vasp"


def plot(*args, **kwargs):
    """Plot the given data, modifying the look with some optional arguments.

    The intent of this function is not to provide a full fledged plotting functionality
    but as a convenient wrapper around the objects used by py4vasp. This gives a
    similar look and feel for the tutorials and facilitates simple plots with a very
    minimal interface. Use a proper plotting library (e.g. matplotlib or plotly) to
    realize more advanced plots.

    Returns
    -------
    Graph
        A graph containing all given series and optional styles.

    Examples
    --------
    Plot simple x-y data with an optional label

    >>> plot(x, y, "label")

    Plot two series in the same graph

    >>> plot((x1, y1), (x2, y2))

    Attributes of the graph are modified by keyword arguments

    >>> plot(x, y, xlabel="xaxis", ylabel="yaxis")
    """
    for_graph = {key: val for key, val in kwargs.items() if key in Graph._fields}
    return Graph(_parse_series(*args, **kwargs), **for_graph)


def _parse_series(*args, **kwargs):
    try:
        return [Series(*arg) for arg in args]
    except TypeError:
        # A TypeError is raised, if plot(x, y) is called instead of plot((x, y)).
        # Because we creating the Series may raise another error, we leave the
        # exception handling first to avoid reraising the TypeError.
        pass
    for_series = {key: val for key, val in kwargs.items() if key in Series._fields}
    return Series(*args, **for_series)


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


@dataclass
class Graph(Sequence):
    """Wraps the functionality to generate graphs of series.

    From a single or multiple series a graph is generated based on the optional
    parameters set in this class.
    """

    series: Series or Sequence[Series]
    "One or more series shown in the graph."
    xlabel: str = None
    "Label for the x axis."
    xticks: dict = None
    "A dictionary specifying positions and labels where ticks are placed on the x axis."
    ylabel: str = None
    "Label for the y axis."
    y2label: str = None
    "Label for the secondary y axis."
    title: str = None
    "Title of the graph."
    _frozen = False

    def __setattr__(self, key, value):
        # prevent adding new attributes to avoid typos, in Python 3.10 this could be
        # handled by setting slots=True when creating the dataclass
        assert not self._frozen or hasattr(self, key)
        super().__setattr__(key, value)

    def __post_init__(self):
        self._frozen = True
        if self._subplot_on:
            if not all(series.subplot for series in self):
                message = "If subplot is used it has to be set for all data in the series and has to be larger 0"
                raise exception.IncorrectUsage(message)
            if len(np.atleast_1d(self.xlabel)) > len(self):
                message = "Subplot was used with more xlabels than number of subplots. Please check your input"
                raise exception.IncorrectUsage(message)
            if len(np.atleast_1d(self.ylabel)) > len(self):
                message = "Subplot was used with more ylabels than number of subplots. Please check your input"
                raise exception.IncorrectUsage(message)

    def __add__(self, other):
        return Graph(tuple(self) + tuple(other), **_merge_fields(self, other))

    def __getitem__(self, index):
        return np.atleast_1d(self.series)[index]

    def __len__(self):
        return np.atleast_1d(self.series).size

    def to_plotly(self):
        "Convert the graph to a plotly figure."
        figure = self._make_plotly_figure()
        for trace, options in self._generate_plotly_traces():
            if options["row"] is None:
                figure.add_trace(trace)
            else:
                figure.add_trace(trace, row=options["row"], col=1)

        return figure

    def show(self):
        "Show the graph with the default look."
        self.to_plotly().show()

    def label(self, new_label):
        """Apply a new label to all series within.

        If there is only a single series, the label will replace the current one. If there
        are more than one, the new label will be prefixed to the existing ones.

        Parameters
        ----------
        new_label : str
            The new label added to the series.
        """
        self.series = [self._make_label(series, new_label) for series in self]
        return self

    def _make_label(self, series, new_label):
        if len(self) > 1:
            new_label = f"{new_label} {series.name}"
        return replace(series, name=new_label)

    def _ipython_display_(self):
        self.to_plotly()._ipython_display_()

    def _generate_plotly_traces(self):
        colors = itertools.cycle(_vasp_colors)
        for series in self:
            if not series.color:
                series = replace(series, color=next(colors))
            yield from series._generate_traces()

    def _make_plotly_figure(self):
        figure = self._figure_with_one_or_two_y_axes()
        self._set_xaxis_options(figure)
        self._set_yaxis_options(figure)
        figure.layout.title.text = self.title
        figure.layout.legend.itemsizing = "constant"
        return figure

    def _figure_with_one_or_two_y_axes(self):
        if self._subplot_on:
            max_row = max(series.subplot for series in self)
            figure = make_subplots(rows=max_row, cols=1)
            figure.update_layout(showlegend=False)
            return figure
        elif any(series.y2 for series in self):
            return make_subplots(specs=[[{"secondary_y": True}]])
        else:
            return go.Figure()

    def _set_xaxis_options(self, figure):
        if self._subplot_on:
            # setting xlabels for subplots
            for row, xlabel in enumerate(np.atleast_1d(self.xlabel)):
                # row indices start @ 1 in plotly
                figure.update_xaxes(title_text=xlabel, row=row + 1, col=1)
        else:
            figure.layout.xaxis.title.text = self.xlabel
        if self.xticks:
            figure.layout.xaxis.tickmode = "array"
            figure.layout.xaxis.tickvals = tuple(self.xticks.keys())
            figure.layout.xaxis.ticktext = self._xtick_labels()

    def _xtick_labels(self):
        # empty labels will be overwritten by plotly so we put a single space in them
        return tuple(label or " " for label in self.xticks.values())

    def _set_yaxis_options(self, figure):
        if self._subplot_on:
            # setting ylabels for subplots
            for row, ylabel in enumerate(np.atleast_1d(self.ylabel)):
                figure.update_yaxes(title_text=ylabel, row=row + 1, col=1)
        else:
            figure.layout.yaxis.title.text = self.ylabel
            if self.y2label:
                figure.layout.yaxis2.title.text = self.y2label

    @property
    def _subplot_on(self):
        return any(series.subplot for series in self)


Graph._fields = tuple(field.name for field in fields(Graph))


def _merge_fields(left_graph, right_graph):
    return {
        field.name: _merge_field(left_graph, right_graph, field.name)
        for field in fields(Graph)
        if field.name != "series"
    }


def _merge_field(left_graph, right_graph, field_name):
    left_field = getattr(left_graph, field_name)
    right_field = getattr(right_graph, field_name)
    if not left_field:
        return right_field
    if not right_field:
        return left_field
    if left_field != right_field:
        message = f"""Cannot combine two graphs with incompatible {field_name}:
    left: {left_field}
    right: {right_field}"""
        raise exception.IncorrectUsage(message)
    return left_field
