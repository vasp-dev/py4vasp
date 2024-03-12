# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._third_party.graph import Contour, Graph, Series


@pytest.fixture
def parabola():
    x = np.linspace(0, 2, 50)
    return Series(x=x, y=x**2, name="parabola")


@pytest.fixture
def sine():
    x = np.linspace(0, np.pi, 50)
    return Series(x=x, y=np.sin(x), name="sine", marker="o")


@pytest.fixture
def two_lines():
    x = np.linspace(0, 3, 30)
    y = np.linspace((0, 4), (1, 3), 30).T
    return Series(x=x, y=y, name="two lines")


@pytest.fixture
def fatband():
    x = np.linspace(-1, 1, 40)
    return Series(x=x, y=np.abs(x), width=x**2, name="fatband")


@pytest.fixture
def two_fatbands():
    x = np.linspace(0, 3, 30)
    y = np.array((x**2, x**3))
    width = np.sqrt(y)
    return Series(x=x, y=y, width=width, name="two fatbands")


@pytest.fixture
def non_numpy():
    x = (1, 2, 3)
    y = (4, 5, 6)
    return Series(x, y), Series(list(x), list(y))


@pytest.fixture
def subplot():
    x1 = np.linspace(0, 10)
    y1 = x1**2
    x2 = np.linspace(-3, 3)
    y2 = x2**3
    return Series(x1, y1, subplot=1), Series(x2, y2, subplot=2)


@pytest.fixture
def rectangle_contour():
    return Contour(
        data=np.linspace(0, 10, 20 * 18).reshape((20, 18)),
        lattice=np.diag([4.0, 3.6]),
        label="cubic contour",
    )


def test_basic_graph(parabola, Assert, not_core):
    graph = Graph(parabola)
    fig = graph.to_plotly()
    assert len(fig.data) == 1
    compare_series(fig.data[0], parabola, Assert)


def test_two_series(parabola, sine, Assert, not_core):
    for graph in (Graph([parabola, sine]), Graph(parabola) + Graph(sine)):
        fig = graph.to_plotly()
        assert len(fig.data) == 2
        for converted, original in zip(fig.data, [parabola, sine]):
            compare_series(converted, original, Assert)


def compare_series(converted, original, Assert):
    Assert.allclose(converted.x, original.x)
    Assert.allclose(converted.y, original.y)
    assert converted.name == original.name


def test_axis_label(parabola, not_core):
    graph = Graph(parabola)
    graph.xlabel = "xaxis label"
    graph.ylabel = "yaxis label"
    fig = graph.to_plotly()
    assert fig.layout.xaxis.title.text == graph.xlabel
    assert fig.layout.yaxis.title.text == graph.ylabel


def test_secondary_yaxis(parabola, sine, Assert, not_core):
    sine.y2 = True
    graph = Graph([parabola, sine])
    graph.y2label = "secondary yaxis label"
    fig = graph.to_plotly()
    assert fig.layout.yaxis2.title.text == graph.y2label
    assert len(fig.data) == 2
    assert fig.data[1].yaxis == "y2"


def check_legend_group(converted, original, first_trace):
    assert converted.legendgroup == original.name
    assert converted.showlegend == first_trace


def test_two_lines(two_lines, Assert, not_core):
    graph = Graph(two_lines)
    fig = graph.to_plotly()
    assert len(fig.data) == 2
    first_trace = True
    for converted, y in zip(fig.data, two_lines.y):
        original = Series(x=two_lines.x, y=y, name=two_lines.name)
        compare_series(converted, original, Assert)
        check_legend_group(converted, original, first_trace)
        if first_trace:
            assert converted.line.color is not None
            color = converted.line.color
        assert converted.line.color == color
        first_trace = False


def compare_series_with_width(converted, original, Assert):
    upper = original.y + original.width
    lower = original.y - original.width
    expected = Series(
        x=np.concatenate((original.x, original.x[::-1])),
        y=np.concatenate((lower, upper[::-1])),
        name=original.name,
    )
    compare_series(converted, expected, Assert)
    assert converted.mode == "none"
    assert converted.fill == "toself"
    assert converted.fillcolor is not None
    assert converted.opacity == 0.5


def test_fatband(fatband, Assert, not_core):
    graph = Graph(fatband)
    fig = graph.to_plotly()
    compare_series_with_width(fig.data[0], fatband, Assert)


def test_two_fatbands(two_fatbands, Assert, not_core):
    graph = Graph(two_fatbands)
    fig = graph.to_plotly()
    assert len(fig.data) == 2
    first_trace = True
    for converted, y, w in zip(fig.data, two_fatbands.y, two_fatbands.width):
        original = Series(x=two_fatbands.x, y=y, width=w, name=two_fatbands.name)
        compare_series_with_width(converted, original, Assert)
        check_legend_group(converted, original, first_trace)
        if first_trace:
            assert converted.fillcolor is not None
            color = converted.fillcolor
        assert converted.fillcolor == color
        first_trace = False


def test_simple_with_marker(sine, not_core):
    graph = Graph(sine)
    fig = graph.to_plotly()
    assert len(fig.data) == 1
    assert fig.data[0].mode == "markers"
    assert fig.data[0].marker.size is None


def test_fatband_with_marker(fatband, Assert, not_core):
    with_marker = dataclasses.replace(fatband, marker="o")
    graph = Graph(with_marker)
    fig = graph.to_plotly()
    assert fig.layout.legend.itemsizing == "constant"
    assert len(fig.data) == 1
    compare_series(fig.data[0], fatband, Assert)
    assert fig.data[0].mode == "markers"
    assert fig.data[0].marker.sizemode == "area"
    Assert.allclose(fig.data[0].marker.size, fatband.width)


def test_two_fatband_with_marker(two_fatbands, Assert, not_core):
    with_marker = dataclasses.replace(two_fatbands, marker="o")
    graph = Graph(with_marker)
    fig = graph.to_plotly()
    assert fig.layout.legend.itemsizing == "constant"
    assert len(fig.data) == 2
    for converted, y, w in zip(fig.data, two_fatbands.y, two_fatbands.width):
        original = Series(x=two_fatbands.x, y=y, name=two_fatbands.name)
        compare_series(converted, original, Assert)
        assert converted.mode == "markers"
        Assert.allclose(converted.marker.size, w)


def test_custom_xticks(parabola, not_core):
    graph = Graph(parabola)
    graph.xticks = {0.1: "X", 0.3: "Y", 0.4: "", 0.8: "Z"}
    fig = graph.to_plotly()
    assert fig.layout.xaxis.tickmode == "array"
    assert fig.layout.xaxis.tickvals == (0.1, 0.3, 0.4, 0.8)
    assert fig.layout.xaxis.ticktext == ("X", "Y", " ", "Z")
    # empty ticks should be replace by " " because otherwise plotly will replace them


def test_title(parabola, not_core):
    graph = Graph(parabola)
    graph.title = "title"
    fig = graph.to_plotly()
    assert fig.layout.title.text == graph.title


def test_merging_of_fields_of_graph(sine, parabola):
    init_all_fields = {field.name: field.name for field in dataclasses.fields(Graph)}
    init_all_fields.pop("series")
    graph1 = Graph(sine, **init_all_fields)
    graph2 = Graph(parabola)
    for field in dataclasses.fields(Graph):
        if field.name == "series":
            continue
        # if only one side is defined, use that one
        graph = graph1 + graph2
        assert getattr(graph, field.name) == field.name
        graph = graph2 + graph1
        assert getattr(graph, field.name) == field.name
        # if both sides are defined, they need to be identical
        graph = graph1 + dataclasses.replace(graph2, **{field.name: field.name})
        assert getattr(graph, field.name) == field.name
        # if they are not an error is raised
        with pytest.raises(exception.IncorrectUsage):
            graph1 + dataclasses.replace(graph2, **{field.name: "other"})
        with pytest.raises(exception.IncorrectUsage):
            dataclasses.replace(graph2, **{field.name: "other"}) + graph1


def test_subplot(subplot, not_core):
    graph = Graph(subplot)
    graph.xlabel = ("first x-axis", "second x-axis")
    graph.ylabel = ("first y-axis", "second y-axis")
    fig = graph.to_plotly()
    assert fig.data[0].xaxis == "x"
    assert fig.data[0].yaxis == "y"
    assert fig.layout.xaxis1.title.text == graph.xlabel[0]
    assert fig.layout.yaxis.title.text == graph.ylabel[0]
    assert fig.data[1].xaxis == "x2"
    assert fig.data[1].yaxis == "y2"
    assert fig.layout.xaxis2.title.text == graph.xlabel[1]
    assert fig.layout.yaxis2.title.text == graph.ylabel[1]


def test_subplot_label_lengths(subplot):
    with pytest.raises(exception.IncorrectUsage):
        Graph(subplot, xlabel=("1", "2", "3"))
    with pytest.raises(exception.IncorrectUsage):
        Graph(subplot, ylabel=("1", "2", "3"))


def test_mixture_subplot_raises_error(parabola, subplot):
    with pytest.raises(exception.IncorrectUsage):
        Graph((parabola,) + subplot)


def test_non_numpy_data(non_numpy, Assert, not_core):
    graph = Graph(non_numpy)
    fig = graph.to_plotly()
    assert len(fig.data) == len(non_numpy)
    for converted, original in zip(fig.data, non_numpy):
        Assert.allclose(converted.x, np.array(original.x))
        Assert.allclose(converted.y, np.array(original.y))


def test_add_label_to_single_line(parabola, Assert):
    graph = Graph(parabola).label("new label")
    assert len(graph.series) == 1
    Assert.allclose(graph.series[0].x, parabola.x)
    Assert.allclose(graph.series[0].y, parabola.y)
    assert graph.series[0].name == "new label"
    assert parabola.name == "parabola"


def test_add_label_to_multiple_lines(parabola, sine, Assert):
    graph = Graph([parabola, sine]).label("new label")
    assert len(graph.series) == 2
    Assert.allclose(graph.series[0].x, parabola.x)
    Assert.allclose(graph.series[0].y, parabola.y)
    assert graph.series[0].name == "new label parabola"
    Assert.allclose(graph.series[1].x, sine.x)
    Assert.allclose(graph.series[1].y, sine.y)
    assert graph.series[1].name == "new label sine"


def test_convert_parabola_to_frame(parabola, Assert, not_core):
    graph = Graph(parabola)
    df = graph.to_frame()
    Assert.allclose(df["parabola.x"], parabola.x)
    Assert.allclose(df["parabola.y"], parabola.y)


def test_convert_sequence_parabola_to_frame(parabola, sine, Assert, not_core):
    sequence = [parabola, sine]
    graph = Graph(sequence)
    df = graph.to_frame()
    Assert.allclose(df["parabola.x"], parabola.x)
    Assert.allclose(df["parabola.y"], parabola.y)
    Assert.allclose(df["sine.x"], sine.x)
    Assert.allclose(df["sine.y"], sine.y)


def test_convert_multiple_lines(two_lines, Assert, not_core):
    graph = Graph(two_lines)
    df = graph.to_frame()
    assert len(df.columns) == 3
    Assert.allclose(df["two_lines.x"], two_lines.x)
    Assert.allclose(df["two_lines.y0"], two_lines.y[0])
    Assert.allclose(df["two_lines.y1"], two_lines.y[1])


def test_convert_two_fatbands_to_frame(two_fatbands, Assert, not_core):
    graph = Graph(two_fatbands)
    df = graph.to_frame()
    Assert.allclose(df["two_fatbands.x"], two_fatbands.x)
    Assert.allclose(df["two_fatbands.y0"], two_fatbands.y[0])
    Assert.allclose(df["two_fatbands.y1"], two_fatbands.y[1])
    Assert.allclose(df["two_fatbands.width0"], two_fatbands.width[0])
    Assert.allclose(df["two_fatbands.width1"], two_fatbands.width[1])


def test_write_csv(tmp_path, two_fatbands, non_numpy, Assert, not_core):
    import pandas as pd

    sequence = [two_fatbands, *non_numpy]
    graph = Graph(sequence)
    graph.to_csv(tmp_path / "filename.csv")
    ref = graph.to_frame()
    actual = pd.read_csv(tmp_path / "filename.csv")
    ref_rounded = np.round(ref.values, 12)
    actual_rounded = np.round(actual.values, 12)
    Assert.allclose(ref_rounded, actual_rounded)


def test_convert_different_length_series_to_frame(
    parabola, two_lines, Assert, not_core
):
    sequence = [two_lines, parabola]
    graph = Graph(sequence)
    df = graph.to_frame()
    assert len(df) == max(len(parabola.x), len(two_lines.x))
    Assert.allclose(df["parabola.x"], parabola.x)
    Assert.allclose(df["parabola.y"], parabola.y)
    pad_width = len(parabola.x) - len(two_lines.x)
    pad_nan = np.repeat(np.nan, pad_width)
    padded_two_lines_x = np.hstack((two_lines.x, pad_nan))
    padded_two_lines_y = np.hstack((two_lines.y, np.vstack((pad_nan, pad_nan))))
    Assert.allclose(df["two_lines.x"], padded_two_lines_x)
    Assert.allclose(df["two_lines.y0"], padded_two_lines_y[0])
    Assert.allclose(df["two_lines.y1"], padded_two_lines_y[1])


@patch("plotly.graph_objs.Figure._ipython_display_")
def test_ipython_display(mock_display, parabola, not_core):
    graph = Graph(parabola)
    graph._ipython_display_()
    mock_display.assert_called_once()


@patch("plotly.graph_objs.Figure.show")
def test_show(mock_show, parabola, not_core):
    graph = Graph(parabola)
    graph.show()
    mock_show.assert_called_once()


def test_nonexisting_attribute_raises_error(parabola):
    with pytest.raises(AssertionError):
        parabola.nonexisting = "not possible"
    graph = Graph(parabola)
    with pytest.raises(AssertionError):
        graph.nonexisting = "not possible"


def test_contour(rectangle_contour, not_core):
    graph = Graph(rectangle_contour)
    fig = graph.to_plotly()
    assert len(fig.data) == 1
