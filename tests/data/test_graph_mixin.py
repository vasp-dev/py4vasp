import pytest
from unittest.mock import MagicMock

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
