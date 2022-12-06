from unittest.mock import MagicMock

import pytest

from py4vasp._data import graph

GRAPH = MagicMock()


class ExampleGraph(graph.Mixin):
    def to_graph(self):
        GRAPH.reset_mock()
        return GRAPH


def test_is_abstract_class():
    with pytest.raises(TypeError):
        graph.Mixin()


def test_plot_wraps_to_graph():
    example = ExampleGraph()
    assert example.plot() == example.to_graph()


def test_converting_graph_to_plotly():
    example = ExampleGraph()
    fig = example.to_plotly()
    GRAPH.to_plotly.assert_called_once_with()
    assert fig == GRAPH.to_plotly.return_value
