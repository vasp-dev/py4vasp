# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp._third_party.graph.graph import Graph
from py4vasp._third_party.graph.series import Series


def plot(x: np.ndarray, y: np.ndarray, label: str = None, **kwargs) -> Graph:
    """Plot data with a simple, clean interface optimized for quick visualization.

    This function provides a streamlined plotting experience with sensible defaults
    and minimal boilerplate. It wraps the :class:`~py4vasp._third_party.graph.graph.Graph`
    and :class:`~py4vasp._third_party.graph.series.Series` classes to enable rapid
    creation of clear, consistent visualizations.

    For simple plots, this function offers an intuitive interface. For more complex
    visualizations, you can extend the returned Graph object or use specialized
    plotting libraries like matplotlib or plotly.

    Parameters
    ----------
    x
        The x coordinates of the data points.
    y
        The y coordinates of the data points.
    label
        Label for the data series, useful for legends.
    **kwargs
        Additional keyword arguments are distributed to :class:`~py4vasp._third_party.graph.series.Series`
        and :class:`~py4vasp._third_party.graph.graph.Graph` to customize appearance and behavior.

    Returns
    -------
    -
        A graph containing the plotted series with the specified styling options.

    Examples
    --------
    Plot simple x-y data with a label

    >>> x = np.array([1, 2, 3])
    >>> y = np.array([4, 5, 6])
    >>> plot(x, y, "my data")
    Graph(series=Series(x=array([1, 2, 3]), y=array([4, 5, 6]), label='my data', ...), ...)

    Combine multiple series in one graph using the + operator

    >>> plot(x, y) + plot(x + 1, y + 2)
    Graph(series=(Series(x=array([1, 2, 3]), y=array([4, 5, 6]), ...), Series(x=array([2, 3, 4]), y=array([6, 7, 8]), ...)), ...)

    Customize axis labels with keyword arguments

    >>> plot(x, y, xlabel="Time (s)", ylabel="Amplitude")
    Graph(series=Series(...), ..., xlabel='Time (s)', ..., ylabel='Amplitude', ...)
    """
    series = _parse_series(x, y, label, **kwargs)
    for_graph = {key: val for key, val in kwargs.items() if key in Graph._fields}
    return Graph(series, **for_graph)


def _parse_series(x, y, label, **kwargs):
    for_series = {key: val for key, val in kwargs.items() if key in Series._fields}
    return Series(x, y, label, **for_series)
