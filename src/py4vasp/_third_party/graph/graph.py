# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Optional

import numpy as np

from py4vasp import exception
from py4vasp._config import VASP_COLORS
from py4vasp._third_party.graph.contour import Contour
from py4vasp._third_party.graph.series import Series
from py4vasp._third_party.graph.trace import Trace
from py4vasp._util import import_

go = import_.optional("plotly.graph_objects")
subplots = import_.optional("plotly.subplots")
pd = import_.optional("pandas")


@dataclass
class Graph(Sequence):
    """A flexible container for creating and managing data visualization graphs.

    The Graph class provides a comprehensive interface for creating, customizing, and
    exporting data visualizations. It supports single or multiple data series, interactive
    plotting with Plotly, and various export formats including CSV and pandas DataFrames.

    This class acts as both a container for data series and a configuration object for
    plot properties such as axis labels, ranges, sizes, and titles. It implements the
    Sequence protocol, allowing iteration and indexing over the contained series.

    Key Features
    ------------
    - Support for single or multiple data series
    - Interactive visualization using Plotly
    - Customizable axis labels, ranges, and tick positions
    - Configurable figure dimensions
    - Subplot support for organizing multiple plots vertically
    - Secondary y-axis support for comparing series with different scales
    - Export capabilities to CSV, pandas DataFrame, and Plotly figures
    - Automatic color cycling for multiple series
    - Contour plot support with aspect ratio handling

    Notes
    -----
    - The class is designed to be immutable after initialization; new attributes cannot
      be added after creation to prevent typos.
    - Graphs can be combined using the + operator, which merges series and reconciles
      compatible settings.
    - When using subplots, all series must have the subplot attribute set to a positive
      integer indicating the subplot row number.
    - Colorbars are automatically positioned to avoid overlap with the legend.
    - For contour plots, axes are hidden and aspect ratios are locked to maintain
      spatial relationships.

    See Also
    --------
    :class:`~py4vasp.graph.Series` : The primary data series type for line and scatter plots.
    :class:`~py4vasp.graph.Contour` : A specialized series type for contour and heatmap visualizations.

    Examples
    --------
    Create a simple graph and modify its properties:

    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> graph = py4vasp.plot(x, y, "my data")
    >>> graph.xlabel = "Time (s)"
    >>> graph.ylabel = "Temperature (K)"
    >>> graph.title = "Temperature vs Time"
    >>> graph.show()

    Modify the axis ranges to zoom into a specific region:

    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x)
    >>> graph = py4vasp.plot(x, y, "sine wave")
    >>> graph.xrange = (2, 8)
    >>> graph.yrange = (-0.5, 0.5)
    >>> graph.show()

    Customize the figure size:

    >>> x = np.array([1, 2, 3])
    >>> y = np.array([4, 5, 6])
    >>> graph = py4vasp.plot(x, y, "data")
    >>> graph.xsize = 1200
    >>> graph.ysize = 800
    >>> graph.show()

    Add custom tick positions and labels:

    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([1, 4, 9, 16])
    >>> graph = py4vasp.plot(x, y, "squares")
    >>> graph.xticks = {0: "start", 1: "one", 2: "two", 3: "end"}
    >>> graph.show()

    Combine multiple series and modify the combined graph:

    >>> x = np.linspace(0, 2*np.pi, 50)
    >>> y1 = np.sin(x)
    >>> y2 = np.cos(x)
    >>> graph1 = py4vasp.plot(x, y1, "sin")
    >>> graph2 = py4vasp.plot(x, y2, "cos")
    >>> combined = graph1 + graph2
    >>> combined.xlabel = "Angle (rad)"
    >>> combined.ylabel = "Amplitude"
    >>> combined.title = "Trigonometric Functions"
    >>> combined.show()
    """

    series: Trace | Sequence[Trace]
    "One or more data series (e.g., Series, Contour, or Trace objects) to be displayed in the graph."
    xlabel: Optional[str] = None
    "Label for the x-axis. For subplots, provide a list of labels corresponding to each subplot."
    xrange: Optional[tuple] = None
    "Tuple specifying the visible range of the x-axis as (min, max)."
    xticks: Optional[dict] = None
    "Dictionary mapping tick positions (keys) to their labels (values) for the x-axis."
    xsize: Optional[int] = 720
    "Width of the figure in pixels."
    ylabel: Optional[str] = None
    "Label for the y-axis. For subplots, provide a list of labels corresponding to each subplot."
    yrange: Optional[tuple] = None
    "Tuple specifying the visible range of the y-axis as (min, max)."
    y2label: Optional[str] = None
    "Label for the secondary y-axis (only applicable when series use the secondary y-axis)."
    ysize: Optional[int] = 540
    "Height of the figure in pixels."
    title: Optional[str] = None
    "Title displayed at the top of the graph."
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

    def to_plotly(self) -> "go.Figure":
        """Convert the graph to a plotly figure for interactive visualization and customization.

        This method transforms the internal graph representation into a Plotly Figure object,
        which enables interactive plotting, customization, and export capabilities. The resulting
        figure includes all traces, shapes, and annotations from the graph, properly organized
        into subplots if multiple rows are specified.

        Once converted to a Plotly figure, you can:
        - Display the graph interactively in Jupyter notebooks or web browsers
        - Further customize the layout, colors, fonts, and other visual properties
        - Export to various formats (HTML, PNG, PDF, SVG)
        - Add additional traces, annotations, or modify existing elements
        - Save the figure for later use or sharing

        Returns
        -------
        go.Figure
            A Plotly Figure object containing all graph elements (traces, shapes, annotations)
            with the configured legend settings.

        Examples
        --------
        >>> # Example 1: Display an interactive plot in a Jupyter notebook
        >>> graph = py4vasp.plot(x=[1, 2, 3], y=[4, 5, 6], label="my data")
        >>> fig = graph.to_plotly()
        >>> fig.show()

        >>> # Example 2: Customize the figure after conversion
        >>> graph = py4vasp.plot(x=[1, 2, 3], y=[4, 5, 6], label="my data")
        >>> fig = graph.to_plotly()
        >>> fig.update_layout(title="Custom Title", template="plotly_dark")
        Figure(...)
        >>> fig.update_xaxes(title_text="Custom X Label")
        Figure(...)
        >>> fig.show()

        >>> # Example 3: Export the figure to an HTML file
        >>> graph = py4vasp.plot(x=[1, 2, 3], y=[4, 5, 6], label="my data")
        >>> fig = graph.to_plotly()
        >>> fig.write_html(path / "my_graph.html")
        """
        figure = self._make_plotly_figure()
        for trace, options in self._generate_plotly_traces():
            if options.get("row") is None:
                figure.add_trace(trace)
            else:
                figure.add_trace(trace, row=options["row"], col=1)
            for shape in options.get("shapes", ()):
                figure.add_shape(**shape)
            for annotation in options.get("annotations", ()):
                figure.add_annotation(**annotation)
        self._set_legend(figure)
        return figure

    def show(self):
        """Display the graph in an interactive viewer.

        This method renders the graph using an interactive plotting backend, allowing
        you to visualize and explore the data. The graph will open in your default
        viewer, typically a web browser or an integrated notebook interface.

        The visualization supports interactive features such as zooming, panning,
        hovering over data points to see values, and toggling legend entries to
        show or hide specific data series.

        Examples
        --------
        Create and display a simple graph:

        >>> x = np.array([1, 2, 3])
        >>> y = np.array([4, 5, 6])
        >>> graph = py4vasp.plot(x, y, "my data")
        >>> graph.show()

        Display a graph after customizing its appearance:

        >>> x = np.array([1, 2, 3])
        >>> y = np.array([4, 5, 6])
        >>> graph = py4vasp.plot(x, y, "my data")
        >>> graph.xlabel = "Time (s)"
        >>> graph.ylabel = "Temperature (K)"
        >>> graph.show()

        In Jupyter notebooks, the graph will be embedded inline, while in scripts
        it will open in a separate browser window.
        """
        self.to_plotly().show()

    def label(self, new_label: str) -> None:
        """Apply a new label to all series within.

        If there is only a single series, the label will replace the current one. If there
        are more than one, the new label will be prefixed to the existing ones.

        Parameters
        ----------
        new_label
            The new label added to the series.

        Examples
        --------
        Replace the current label with a new one for a single series.

        >>> x = np.array([1, 2, 3])
        >>> y = np.array([4, 5, 6])
        >>> graph = py4vasp.plot(x, y, "old label")
        >>> graph.label("new label")
        Graph(series=[Series(..., label='new label', ...)], ...)

        Prefix the current label with a new one for multiple series.

        >>> x = np.array([1, 2, 3])
        >>> y = np.array([4, 5, 6])
        >>> graph = py4vasp.plot(x, y, "one") + py4vasp.plot(x, y, "two")
        >>> graph.label("prefix")
        Graph(series=[Series(..., label='prefix one', ...), Series(..., label='prefix two', ...)], ...)
        """
        self.series = [self._make_label(series, new_label) for series in self]
        return self

    def _make_label(self, series, new_label):
        if len(self) > 1:
            new_label = f"{new_label} {series.label}"
        return replace(series, label=new_label)

    def _ipython_display_(self):
        self.to_plotly()._ipython_display_()

    def _generate_plotly_traces(self):
        colors_without_dark = (
            color for name, color in VASP_COLORS.items() if name != "dark"
        )
        colors = itertools.cycle(colors_without_dark)
        for series in self:
            series = _set_color_if_not_present(series, colors)
            yield from series.to_plotly()

    def _make_plotly_figure(self):
        figure = self._figure_with_one_or_two_y_axes()
        self._set_xaxis_options(figure)
        self._set_yaxis_options(figure)
        figure.layout.title.text = self.title
        if self.xsize:
            figure.layout.width = self.xsize
        figure.layout.height = self.ysize
        figure.layout.legend.itemsizing = "constant"
        return figure

    def _figure_with_one_or_two_y_axes(self):
        has_secondary_y_axis = lambda series: isinstance(series, Series) and series.y2
        if self._subplot_on:
            max_row = max(series.subplot for series in self)
            figure = subplots.make_subplots(rows=max_row, cols=1)
            figure.update_layout(showlegend=False)
            return figure
        elif any(has_secondary_y_axis(series) for series in self):
            return subplots.make_subplots(specs=[[{"secondary_y": True}]])
        else:
            return go.Figure()

    def _set_legend(self, figure):
        default_colorbar_x = 1.02
        colorbar_spacing = 0.18
        subplot_colorbars = defaultdict(list)

        # Step 1: Group colorbars by subplot (based on axis mapping)
        for trace in figure.data:
            if hasattr(trace, "colorbar") and trace.colorbar:
                xaxis = getattr(trace, "xaxis", "x")
                yaxis = getattr(trace, "yaxis", "y")
                subplot_key = f"{xaxis}_{yaxis}"
                subplot_colorbars[subplot_key].append(trace)

        max_colorbar_x = 1.0  # Track global maximum x for legend placement

        # Step 2: Assign x-positions to colorbars, per subplot
        for colorbar_list in subplot_colorbars.values():
            for i, trace in enumerate(colorbar_list):
                cb = trace.colorbar
                # If user hasn't explicitly set `x`, place them with spacing
                if not hasattr(cb, "x") or cb.x is None:
                    cb.x = default_colorbar_x + i * colorbar_spacing
                max_colorbar_x = max(max_colorbar_x, cb.x)

        # Step 3: Place legend safely to the right of all colorbars
        if max_colorbar_x > 1.0:
            legend_x = max_colorbar_x + colorbar_spacing
            figure.update_layout(
                legend=dict(x=legend_x, y=1.0, xanchor="left", yanchor="top")
            )

        figure.update_layout(margin=dict(r=120))

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
        if self.xrange:
            figure.layout.xaxis.range = self.xrange
        if self._all_are_contour():
            figure.layout.xaxis.visible = False

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
        if self.yrange:
            figure.layout.yaxis.range = self.yrange
        if self._all_are_contour():
            figure.layout.yaxis.visible = False
        if self._any_are_contour():
            figure.layout.yaxis.scaleanchor = "x"

    def _all_are_contour(self):
        return all(isinstance(series, Contour) for series in self)

    def _any_are_contour(self):
        return any(isinstance(series, Contour) for series in self)

    def to_frame(self) -> "pd.DataFrame":
        """Convert graph to a pandas dataframe.

        Every series will have at least two columns, named after the series name
        with the suffix x and y. Additionally, if weights are provided, they will
        also be written out as another column. If a series does not have a name, a
        name will be generated based on a uuid.

        Returns
        -------
        -
            A pandas dataframe with columns for each series in the graph

        Examples
        --------
        Convert a graph with a single series to a dataframe:

        >>> graph = py4vasp.plot(x=[1, 2, 3], y=[4, 5, 6], label="data")
        >>> df = graph.to_frame()
        >>> print(df)
           data.x  data.y
        0       1       4
        1       2       5
        2       3       6

        Convert a graph with multiple series to a dataframe:

        >>> graph = Graph(series=[
        ...     Series(x=[1, 2], y=[3, 4], label="series1"),
        ...     Series(x=[1, 2], y=[5, 6], label="series2")
        ... ])
        >>> df = graph.to_frame()
        >>> print(df)
           series1.x  series1.y  series2.x  series2.y
        0          1          3          1          5
        1          2          4          2          6

        Convert a graph with weighted series to a dataframe:

        >>> graph = Graph(series=Series(x=[1, 2], y=[3, 4], weight=[0.5, 0.8], label="weighted"))
        >>> df = graph.to_frame()
        >>> print(df)
           weighted.x  weighted.y  weighted.weight
        0           1           3              0.5
        1           2           4              0.8
        """
        df = pd.DataFrame()
        for series in np.atleast_1d(self.series):
            _df = self._create_and_populate_df(series)
            df = df.join(_df, how="outer")
        return df

    def to_csv(self, filename: str | Path) -> None:
        """Export graph data to a CSV file.

        This method saves all series data in the graph to a CSV file. Each series
        will have columns for x and y values, named after the series label. If weights
        are provided, they will also be included as additional columns.

        Parameters
        ----------
        filename
            Path to the output CSV file.

        Examples
        --------
        Export a simple graph to CSV:

        >>> x = np.array([1, 2, 3])
        >>> y = np.array([4, 5, 6])
        >>> graph = py4vasp.plot(x, y, "my data")
        >>> graph.to_csv(str(path / "output.csv"))
        >>> with open(path / "output.csv") as f:
        ...     print(f.read())
        my_data.x,my_data.y
        1,4
        2,5
        3,6

        Export a graph with multiple series:

        >>> x = np.array([1, 2, 3])
        >>> y1 = np.array([4, 5, 6])
        >>> y2 = np.array([7, 8, 9])
        >>> graph = py4vasp.plot(x, y1, "series 1") + py4vasp.plot(x, y2, "series 2")
        >>> graph.to_csv(path / "multi_series.csv")
        >>> with open(path / "multi_series.csv") as f:
        ...     print(f.read())
        series_1.x,series_1.y,series_2.x,series_2.y
        1,4,1,7
        2,5,2,8
        3,6,3,9

        Export a graph with weighted data:

        >>> from py4vasp._third_party.graph.series import Series
        >>> x = np.array([1, 2, 3])
        >>> y = np.array([4, 5, 6])
        >>> weight = np.array([0.1, 0.2, 0.3])
        >>> series = Series(x=x, y=y, weight=weight, label="weighted data")
        >>> graph = Graph(series=series)
        >>> graph.to_csv(path / "weighted_output.csv")
        >>> with open(path / "weighted_output.csv") as f:
        ...     print(f.read())
        weighted_data.x,weighted_data.y,weighted_data.weight
        1,4,0.1
        2,5,0.2
        3,6,0.3
        """
        df = self.to_frame()
        df.to_csv(filename, index=False)

    def _create_and_populate_df(self, series):
        df = pd.DataFrame()
        df[self._name_column(series, "x", None)] = series.x
        for idx, series_y in enumerate(np.atleast_2d(series.y)):
            df[self._name_column(series, "y", idx)] = series_y
        if series.weight is not None:
            assert series.weight.ndim == series.y.ndim
            for idx, series_weight in enumerate(np.atleast_2d(series.weight)):
                df[self._name_column(series, "weight", idx)] = series_weight
        return df

    def _name_column(self, series, suffix, idx=None):
        if series.label:
            text_suffix = series.label.replace(" ", "_") + f".{suffix}"
        else:
            text_suffix = "series_" + str(uuid.uuid1())
        if series.y.ndim == 1 or idx is None:
            return text_suffix
        else:
            return f"{text_suffix}{idx}"

    @property
    def _subplot_on(self):
        has_subplot = lambda series: isinstance(series, Series) and series.subplot
        return any(has_subplot(series) for series in self)


def _set_color_if_not_present(series, color_iterator):
    if isinstance(series, Contour):
        return series
    if not series.color:
        series = replace(series, color=next(color_iterator))
    return series


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
