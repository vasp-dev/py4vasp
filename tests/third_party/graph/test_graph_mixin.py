# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from py4vasp import exception
from py4vasp._third_party import graph

GRAPH = MagicMock(spec=graph.Graph)


class ExampleGraph(graph.Mixin):
    _path = Path("folder")

    def to_graph(self):
        GRAPH.reset_mock()
        return GRAPH


@pytest.fixture(autouse=True)
def existing_output_directory(tmp_path, monkeypatch):
    # the files are written to the path of the quantity, which must exist
    monkeypatch.setattr(ExampleGraph, "_path", tmp_path)
    monkeypatch.setattr(WithArguments, "_path", tmp_path)


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


def test_convert_graph_to_frame():
    example = ExampleGraph()
    df = example.to_frame()
    GRAPH.to_frame.assert_called_once_with()
    assert df == GRAPH.to_frame.return_value


def test_convert_graph_to_csv():
    example = ExampleGraph()
    example.to_csv()
    GRAPH.to_frame.assert_called_once_with()
    full_path = example._path / "example_graph.csv"
    df = GRAPH.to_frame.return_value
    df.to_csv.assert_called_once_with(full_path, index=False)


def test_converting_graph_to_csv_expands_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    example = ExampleGraph()
    example.to_csv(filename="~/example.csv")
    df = GRAPH.to_frame.return_value
    df.to_csv.assert_called_once_with(home / "example.csv", index=False)


def test_converting_graph_to_csv_rejects_missing_directory():
    example = ExampleGraph()
    GRAPH.reset_mock()
    with pytest.raises(exception.FileAccessError):
        example.to_csv(filename="missing/example.csv")
    GRAPH.to_frame.assert_not_called()


def test_converting_graph_to_image():
    example = ExampleGraph()
    example.to_image()
    GRAPH.to_image.assert_called_once_with(example._path / "example_graph.png")


def test_converting_graph_to_csv_with_relative_filename():
    example = ExampleGraph()
    example.to_csv(filename="example.csv")
    full_path = example._path / "example.csv"
    GRAPH.to_frame.assert_called_once_with()
    df = GRAPH.to_frame.return_value
    df.to_csv.assert_called_once_with(full_path, index=False)


def test_converting_graph_to_csv_with_absolute_filename():
    example = ExampleGraph()
    basedir_path = example._path.absolute()
    full_path = basedir_path / "example.csv"
    example.to_csv(filename=full_path)
    GRAPH.to_frame.assert_called_once_with()
    df = GRAPH.to_frame.return_value
    df.to_csv.assert_called_once_with(full_path, index=False)


def test_converting_graph_to_image_with_filename():
    example = ExampleGraph()
    example.to_image(filename="example.jpg")
    GRAPH.to_image.assert_called_once_with(example._path / "example.jpg")


def test_converting_graph_to_image_with_absolute_filename(tmp_path):
    example = ExampleGraph()
    full_path = tmp_path / "elsewhere" / "example.jpg"
    full_path.parent.mkdir()
    example.to_image(filename=full_path)
    GRAPH.to_image.assert_called_once_with(full_path)


def test_converting_graph_to_image_expands_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    example = ExampleGraph()
    example.to_image(filename="~/example.png")
    GRAPH.to_image.assert_called_once_with(home / "example.png")


def test_converting_graph_to_image_rejects_missing_directory():
    example = ExampleGraph()
    GRAPH.reset_mock()
    with pytest.raises(exception.FileAccessError):
        example.to_image(filename="missing/example.png")
    GRAPH.to_image.assert_not_called()


@pytest.mark.parametrize("filename", ["example", "example.txt", "example.eps"])
def test_converting_graph_to_image_rejects_unsupported_extension(filename):
    example = ExampleGraph()
    GRAPH.reset_mock()
    with pytest.raises(exception.IncorrectUsage):
        example.to_image(filename=filename)
    GRAPH.to_image.assert_not_called()


@pytest.mark.parametrize(
    "method, filename",
    [
        ("to_image", "example.jpg"),
        ("to_image", Path("example.PNG")),
        ("to_csv", "a.csv"),
    ],
)
def test_filename_passed_positionally_points_to_the_keyword(method, filename):
    # Graph.to_image takes the filename positionally, so users try that here, too;
    # here the positional arguments select what is plotted
    example = WithArguments()
    GRAPH.reset_mock()
    with pytest.raises(exception.IncorrectUsage, match="filename="):
        getattr(example, method)(filename)
    GRAPH.to_plotly.assert_not_called()
    GRAPH.to_frame.assert_not_called()


class WithArguments(graph.Mixin):
    _path = Path("/absolute/path")

    def to_graph(self, mandatory, optional=None):
        GRAPH.reset_mock()
        GRAPH.arguments = {"mandatory": mandatory, "optional": optional}
        return GRAPH


def test_arguments_passed_by_plot():
    example = WithArguments()
    graph = example.plot("only mandatory")
    assert graph.arguments == {"mandatory": "only mandatory", "optional": None}
    graph = example.plot("first", "second")
    assert graph.arguments == {"mandatory": "first", "optional": "second"}
    graph = example.plot(optional="foo", mandatory="bar")
    assert graph.arguments == {"mandatory": "bar", "optional": "foo"}


def test_arguments_passed_by_to_plotly():
    example = WithArguments()
    example.to_plotly("only mandatory")
    assert GRAPH.arguments == {"mandatory": "only mandatory", "optional": None}
    example.to_plotly("first", "second")
    assert GRAPH.arguments == {"mandatory": "first", "optional": "second"}
    example.to_plotly(optional="foo", mandatory="bar")
    assert GRAPH.arguments == {"mandatory": "bar", "optional": "foo"}


def test_arguments_passed_by_to_image():
    example = WithArguments()
    example.to_image("only mandatory")
    assert GRAPH.arguments == {"mandatory": "only mandatory", "optional": None}
    example.to_image("first", "second", filename="example.png")
    assert GRAPH.arguments == {"mandatory": "first", "optional": "second"}
    example.to_image(optional="foo", mandatory="bar")
    assert GRAPH.arguments == {"mandatory": "bar", "optional": "foo"}


class MultipleGraphs(graph.Mixin):
    _path = Path("./relative_path")

    def to_graph(self):
        series = [graph.Series([1], [2], "foo"), graph.Series([3], [4], "bar")]
        return {
            "first": graph.Graph(graph.Series([5], [6], "ignored")),
            "second": graph.Graph(series),
        }


def test_multiple_graphs_merged_by_plot():
    example = MultipleGraphs()
    result = example.plot()
    assert len(result) == 3
    assert result[0].label == "first"
    assert result[1].label == "second foo"
    assert result[2].label == "second bar"
