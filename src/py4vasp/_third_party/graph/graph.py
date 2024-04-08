# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace

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
    """Wraps the functionality to generate graphs of series.

    From a single or multiple series a graph is generated based on the optional
    parameters set in this class.
    """

    series: Trace or Sequence[Trace]
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
            if options.get("row") is None:
                figure.add_trace(trace)
            else:
                figure.add_trace(trace, row=options["row"], col=1)
            for shape in options.get("shapes", ()):
                figure.add_shape(**shape)
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
        colors = itertools.cycle(VASP_COLORS)
        for series in self:
            series = _set_color_if_not_present(series, colors)
            yield from series.to_plotly()

    def _make_plotly_figure(self):
        figure = self._figure_with_one_or_two_y_axes()
        self._set_xaxis_options(figure)
        self._set_yaxis_options(figure)
        figure.layout.title.text = self.title
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
        if self._all_are_contour():
            figure.layout.yaxis.visible = False
        if self._any_are_contour():
            figure.layout.yaxis.scaleanchor = "x"
            figure.layout.height = 500

    def _all_are_contour(self):
        return all(isinstance(series, Contour) for series in self)

    def _any_are_contour(self):
        return any(isinstance(series, Contour) for series in self)

    def to_frame(self):
        """Convert graph to a pandas dataframe.

        Every series will have at least two columns, named after the series name
        with the suffix x and y. Additionally, if weights are provided, they will
        also be written out as another column. If a series does not have a name, a
        name will be generated based on a uuid.

        Returns
        -------
        Dataframe
            A pandas dataframe with columns for each series in the graph
        """
        df = pd.DataFrame()
        for series in np.atleast_1d(self.series):
            _df = self._create_and_populate_df(series)
            df = df.join(_df, how="outer")
        return df

    def to_csv(self, filename):
        """Export graph to a csv file.

        Starting from the dataframe generated from `to_frame`, use the `to_csv` method
        implemented in pandas to write out a csv file with a given filename

        Parameters
        ----------
        filename: str | Path
            Name of the exported csv file
        """
        df = self.to_frame()
        df.to_csv(filename, index=False)

    def _create_and_populate_df(self, series):
        df = pd.DataFrame()
        df[self._name_column(series, "x", None)] = series.x
        for idx, series_y in enumerate(np.atleast_2d(series.y)):
            df[self._name_column(series, "y", idx)] = series_y
        if series.width is not None:
            assert series.width.ndim == series.y.ndim
            for idx, series_width in enumerate(np.atleast_2d(series.width)):
                df[self._name_column(series, "width", idx)] = series_width
        return df

    def _name_column(self, series, suffix, idx=None):
        if series.name:
            text_suffix = series.name.replace(" ", "_") + f".{suffix}"
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
