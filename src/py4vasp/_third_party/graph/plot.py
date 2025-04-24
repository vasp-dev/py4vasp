# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._third_party.graph.graph import Graph
from py4vasp._third_party.graph.series import Series


def plot(x, y, label=None, **kwargs):
    """Plot the given data, modifying the look with some optional arguments.

    The intent of this function is not to provide a full fledged plotting functionality
    but as a convenient wrapper around the objects used by py4vasp. This gives a
    similar look and feel for the tutorials and facilitates simple plots with a very
    minimal interface. Use a proper plotting library (e.g. matplotlib or plotly) to
    realize more advanced plots.

    Parameters
    ----------
    x : np.ndarray
        The x values of the coordinates.
    y : np.ndarray
        The y values of the coordinates.
    label : str
        If set this will be used to label the series.
    **kwargs
        All additional arguments will be passed to initialize Series and Graph.

    Returns
    -------
    Graph
        A graph containing all given series and optional styles.

    Examples
    --------
    Plot simple x-y data with an optional label

    >>> plot(x, y, "label")

    Plot two series in the same graph

    >>> plot(x1, y1) + plot(x2, y2)

    Attributes of the graph are modified by keyword arguments

    >>> plot(x, y, xlabel="xaxis", ylabel="yaxis")
    """
    series = _parse_series(x, y, label, **kwargs)
    for_graph = {key: val for key, val in kwargs.items() if key in Graph._fields}
    return Graph(series, **for_graph)


def _parse_series(x, y, label, **kwargs):
    for_series = {key: val for key, val in kwargs.items() if key in Series._fields}
    return Series(x, y, label, **for_series)
