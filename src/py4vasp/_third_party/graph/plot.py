# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._third_party.graph.graph import Graph
from py4vasp._third_party.graph.series import Series


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
