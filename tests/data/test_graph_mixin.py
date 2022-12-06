import pytest

from py4vasp._data import graph


class ExampleGraph(graph.Mixin):
    pass


def test_is_abstract_class():
    with pytest.raises(TypeError):
        graph.Mixin()
