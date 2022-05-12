from py4vasp._third_party.graph import Graph, Series, plot
import py4vasp.exceptions as exception
import numpy as np
import pytest
from unittest.mock import patch


@pytest.fixture
def parabola():
    x = np.linspace(0, 2, 50)
    return Series(x=x, y=x ** 2, name="parabola")


@pytest.fixture
def sine():
    x = np.linspace(0, np.pi, 50)
    return Series(x=x, y=np.sin(x), name="sine")


@pytest.fixture
def two_lines():
    x = np.linspace(0, 3, 30)
    y = np.linspace((0, 4), (1, 3), 30).T
    return Series(x=x, y=y, name="two lines")


@pytest.fixture
def fatband():
    x = np.linspace(-1, 1, 40)
    return Series(x=x, y=np.abs(x), width=x ** 2, name="fatband")


@pytest.fixture
def non_numpy():
    x = (1, 2, 3)
    y = (4, 5, 6)
    return Series(x, y), Series(list(x), list(y))


def test_basic_graph(parabola, Assert):
    graph = Graph(parabola)
    fig = graph.to_plotly()
    assert len(fig.data) == 1
    compare_series(fig.data[0], parabola, Assert)


def test_two_series(parabola, sine, Assert):
    graph = Graph([parabola, sine])
    fig = graph.to_plotly()
    assert len(fig.data) == 2
    for converted, original in zip(fig.data, [parabola, sine]):
        compare_series(converted, original, Assert)


def compare_series(converted, original, Assert):
    Assert.allclose(converted.x, original.x)
    Assert.allclose(converted.y, original.y)
    assert converted.name == original.name


def test_axis_label(parabola):
    graph = Graph(parabola)
    graph.xlabel = "xaxis label"
    graph.ylabel = "yaxis label"
    fig = graph.to_plotly()
    assert fig.layout.xaxis.title.text == graph.xlabel
    assert fig.layout.yaxis.title.text == graph.ylabel


def test_secondary_yaxis(parabola, sine, Assert):
    sine.y2 = True
    graph = Graph([parabola, sine])
    graph.y2label = "secondary yaxis label"
    fig = graph.to_plotly()
    assert fig.layout.yaxis2.title.text == graph.y2label
    assert len(fig.data) == 2
    assert fig.data[1].yaxis == "y2"


def test_two_lines(two_lines, Assert):
    graph = Graph(two_lines)
    fig = graph.to_plotly()
    assert len(fig.data) == 2
    first_trace = True
    for converted, y in zip(fig.data, two_lines.y):
        original = Series(x=two_lines.x, y=y, name=two_lines.name)
        compare_series(converted, original, Assert)
        assert converted.legendgroup == original.name
        assert converted.showlegend == first_trace
        if first_trace:
            assert converted.line.color is not None
            color = converted.line.color
        assert converted.line.color == color
        first_trace = False


def test_fatband(fatband, Assert):
    graph = Graph(fatband)
    fig = graph.to_plotly()
    upper = fatband.y + fatband.width
    lower = fatband.y - fatband.width
    expected = Series(
        x=np.concatenate((fatband.x, fatband.x[::-1])),
        y=np.concatenate((lower, upper[::-1])),
        name=fatband.name,
    )
    assert len(fig.data) == 1
    compare_series(fig.data[0], expected, Assert)
    assert fig.data[0].mode == "none"
    assert fig.data[0].fill == "toself"
    assert fig.data[0].fillcolor is not None
    assert fig.data[0].opacity == 0.5


def test_custom_xticks(parabola):
    graph = Graph(parabola)
    graph.xticks = {0.1: "X", 0.3: "Y", 0.4: "", 0.8: "Z"}
    fig = graph.to_plotly()
    assert fig.layout.xaxis.tickmode == "array"
    assert fig.layout.xaxis.tickvals == (0.1, 0.3, 0.4, 0.8)
    assert fig.layout.xaxis.ticktext == ("X", "Y", " ", "Z")
    # empty ticks should be replace by " " because otherwise plotly will replace them


def test_title(parabola):
    graph = Graph(parabola)
    graph.title = "title"
    fig = graph.to_plotly()
    assert fig.layout.title.text == graph.title


def test_non_numpy_data(non_numpy, Assert):
    graph = Graph(non_numpy)
    fig = graph.to_plotly()
    assert len(fig.data) == len(non_numpy)
    for converted, original in zip(fig.data, non_numpy):
        Assert.allclose(converted.x, np.array(original.x))
        Assert.allclose(converted.y, np.array(original.y))


@patch("plotly.graph_objs.Figure._ipython_display_")
def test_ipython_display(mock_display, parabola):
    graph = Graph(parabola)
    graph._ipython_display_()
    mock_display.assert_called_once()


@patch("plotly.graph_objs.Figure.show")
def test_show(mock_show, parabola):
    graph = Graph(parabola)
    graph.show()
    mock_show.assert_called_once()


def test_plot():
    x1, x2, y1, y2 = np.random.random((4, 50))
    series0 = Series(x1, y1)
    series1 = Series(x1, y1, "label1")
    series2 = Series(x2, y2, "label2")
    assert plot(x1, y1) == Graph(series0)
    assert plot(x1, y1, "label1") == Graph(series1)
    assert plot(x1, y1, name="label1") == Graph(series1)
    assert plot(x1, y1, xlabel="xaxis") == Graph(series0, xlabel="xaxis")
    assert plot((x1, y1)) == Graph([series0])
    assert plot((x1, y1), (x2, y2, "label2")) == Graph([series0, series2])
    assert plot((x1, y1), xlabel="xaxis") == Graph([series0], xlabel="xaxis")


def test_plot_small_dataset():
    for length in range(10):
        x = np.linspace(0, 1, length)
        y = x ** 2
        series = Series(x, y)
        assert plot(x, y) == Graph(series)


def test_plot_inconsistent_length():
    x = np.zeros(10)
    y = np.zeros(20)
    with pytest.raises(exception.IncorrectUsage):
        plot(x, y)
    with pytest.raises(exception.IncorrectUsage):
        plot((x, y))
