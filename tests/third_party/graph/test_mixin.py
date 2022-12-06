# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from py4vasp._third_party import graph

GRAPH = MagicMock()


class ExampleGraph(graph.Mixin):
    _path = Path("folder")

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


def test_converting_graph_to_image():
    example = ExampleGraph()
    example.to_image()
    fig = GRAPH.to_plotly.return_value
    fig.write_image.assert_called_once_with(example._path / "example_graph.png")


def test_converting_graph_to_image_with_filename():
    example = ExampleGraph()
    example.to_image(filename="example.jpg")
    fig = GRAPH.to_plotly.return_value
    fig.write_image.assert_called_once_with(example._path / "example.jpg")


def test_filename_is_keyword_only_argument():
    example = ExampleGraph()
    with pytest.raises(TypeError):
        example.to_image("example.jpg")


class ExampleWithArguments(graph.Mixin):
    _path = Path("/absolute/path")

    def to_graph(self, mandatory, optional=None):
        GRAPH.reset_mock()
        GRAPH.arguments = {"mandatory": mandatory, "optional": optional}
        GRAPH.optional = optional
        return GRAPH


def test_arguments_passed_by_plot():
    example = ExampleWithArguments()
    graph = example.plot("only mandatory")
    assert graph.arguments == {"mandatory": "only mandatory", "optional": None}
    graph = example.plot("first", "second")
    assert graph.arguments == {"mandatory": "first", "optional": "second"}
    graph = example.plot(optional="foo", mandatory="bar")
    assert graph.arguments == {"mandatory": "bar", "optional": "foo"}


def test_arguments_passed_by_to_plotly():
    example = ExampleWithArguments()
    example.to_plotly("only mandatory")
    assert GRAPH.arguments == {"mandatory": "only mandatory", "optional": None}
    example.to_plotly("first", "second")
    assert GRAPH.arguments == {"mandatory": "first", "optional": "second"}
    example.to_plotly(optional="foo", mandatory="bar")
    assert GRAPH.arguments == {"mandatory": "bar", "optional": "foo"}


def test_arguments_passed_by_to_image():
    example = ExampleWithArguments()
    example.to_image("only mandatory")
    assert GRAPH.arguments == {"mandatory": "only mandatory", "optional": None}
    example.to_image("first", "second", filename="example.png")
    assert GRAPH.arguments == {"mandatory": "first", "optional": "second"}
    example.to_image(optional="foo", mandatory="bar")
    assert GRAPH.arguments == {"mandatory": "bar", "optional": "foo"}
