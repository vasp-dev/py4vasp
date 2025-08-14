# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import re
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import _config, exception
from py4vasp._third_party.graph import Contour, Graph, Series
from py4vasp._util import import_, slicing

px = import_.optional("plotly.express")


@pytest.fixture
def parabola():
    x = np.linspace(0, 2, 50)
    return Series(x=x, y=x**2, label="parabola")


@pytest.fixture
def sine():
    x = np.linspace(0, np.pi, 50)
    return Series(x=x, y=np.sin(x), label="sine", marker="o")


@pytest.fixture
def two_lines():
    x = np.linspace(0, 3, 30)
    y = np.linspace((0, 4), (1, 3), 30).T
    return Series(x=x, y=y, label="two lines")


@pytest.fixture
def fatband():
    x = np.linspace(-1, 1, 40)
    return Series(x=x, y=np.abs(x), weight=x**2, label="fatband")


@pytest.fixture
def two_fatbands():
    x = np.linspace(0, 3, 30)
    y = np.array((x**2, x**3))
    weight = np.sqrt(y)
    return Series(x=x, y=y, weight=weight, label="two fatbands")


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
        lattice=slicing.Plane(np.diag([4.0, 3.6]), cut="c"),
        label="rectangle contour",
        isolevels=True,
    )


@pytest.fixture
def tilted_contour():
    return Contour(
        data=np.linspace(0, 5, 16 * 20).reshape((16, 20)),
        lattice=slicing.Plane(np.array([[2, 3], [2, -3]]), cut="b"),
        label="tilted contour",
        supercell=(2, 1),
        show_cell=False,
    )


@pytest.fixture
def simple_quiver():
    return Contour(
        data=np.array([[(y, x) for x in range(3)] for y in range(5)]).T,
        lattice=slicing.Plane(np.diag((3, 5)), cut="a"),
        label="quiver plot",
        scale_arrows=1.7,
    )


@pytest.fixture
def dense_quiver():
    return Contour(
        data=np.linspace(-4, 2, 2 * 41 * 25).reshape((2, 41, 25)),
        lattice=slicing.Plane(np.diag((4, 2)), cut="b"),
        supercell=(2, 3),
        label="quiver plot",
        scale_arrows=0.5,
    )


@pytest.fixture
def complex_quiver():
    return Contour(
        data=np.linspace(-3, 3, 2 * 12 * 10).reshape((2, 12, 10)),
        lattice=slicing.Plane([[3, 2], [-3, 2]]),  # cut not set
        label="quiver plot",
        supercell=(3, 2),
    )


@pytest.mark.parametrize(
    "color_scheme",
    [
        "monochrome",
        "auto",
        "sequential",
        "diverging",
        "positive",
        "negative",
    ],
)
def test_contour_color_scheme(color_scheme, not_core):
    contour = Contour(
        data=np.linspace(0, 10, 20 * 18).reshape((20, 18)),
        lattice=slicing.Plane(np.diag([4.0, 3.6]), cut="c"),
        label="rectangle contour",
        color_scheme=color_scheme,
        isolevels=True,
    )
    assert color_scheme == contour.color_scheme
    graph = Graph(contour)
    fig = graph.to_plotly()
    expected_colorscales_dict = {
        "default": [
            (0, _config.VASP_COLORS["blue"]),
            (0.5, "white"),
            (1, _config.VASP_COLORS["red"]),
        ],
        "auto": px.colors.sequential.Reds,
        "positive": px.colors.sequential.Reds,
        "negative": px.colors.sequential.Blues_r,
        "sequential": px.colors.sequential.Viridis,
        "diverging": px.colors.diverging.RdBu_r,
        "monochrome": px.colors.sequential.turbid_r,
    }
    expected_colorscale = expected_colorscales_dict.get(
        color_scheme, expected_colorscales_dict.get("default", None)
    )
    try:
        assert (fig.data[0].colorscale == expected_colorscale) or (
            [s[1] for s in fig.data[0].colorscale] == expected_colorscale
        )
    except AssertionError:
        raise NotImplementedError(
            f"Color scheme {color_scheme} diverges from expectation:"
            + "\nExpected vs. actual:\n"
            + "\n".join(
                [
                    "\t" + f"{expect_cs}" + "\t" + f"{actual_cs}" + "\n"
                    for expect_cs, actual_cs in zip(
                        expected_colorscale, fig.data[0].colorscale
                    )
                ]
            )
        )


@pytest.mark.xfail
@pytest.mark.parametrize("contrast_mode", [True, False])
def test_contour_contrast_mode(contrast_mode, not_core):
    contour = Contour(
        data=np.linspace(0, 10, 20 * 18).reshape((20, 18)),
        lattice=slicing.Plane(np.diag([4.0, 3.6]), cut="c"),
        label="rectangle contour",
        contrast_mode=contrast_mode,
        isolevels=True,
    )
    assert contrast_mode == contour.contrast_mode
    graph = Graph(contour)
    raise NotImplementedError("Implementation not yet complete.")


@pytest.mark.parametrize(
    "color_limits", [None, (None, None), (-9, None), (None, 9), (-11, 11)]
)
def test_contour_color_limits(color_limits, not_core):
    contour = Contour(
        data=np.linspace(-10, 10, 20 * 18).reshape((20, 18)),
        lattice=slicing.Plane(np.diag([4.0, 3.6]), cut="c"),
        label="rectangle contour",
        color_limits=color_limits,
        isolevels=True,
    )
    assert color_limits == contour.color_limits
    graph = Graph(contour)
    expected_zmin = -10
    expected_zmax = 10
    if color_limits is not None:
        _zmin, _zmax = color_limits
        if _zmin is not None:
            expected_zmin = _zmin
        if _zmax is not None:
            expected_zmax = _zmax
    fig = graph.to_plotly()
    assert fig.data[0].zmin == expected_zmin
    assert fig.data[0].zmax == expected_zmax


@pytest.mark.parametrize("units", ["", None, "px", "m/s"])
def test_contour_colorbar_label(units, not_core):
    contour = Contour(
        data=np.linspace(0, 10, 20 * 18).reshape((20, 18)),
        lattice=slicing.Plane(np.diag([4.0, 3.6]), cut="c"),
        label="rectangle contour",
        colorbar_label=units,
        isolevels=True,
    )
    assert units == contour.colorbar_label
    graph = Graph(contour)
    fig = graph.to_plotly()
    if units != "":
        assert fig.data[0].colorbar.title.text == units
    else:
        assert fig.data[0].colorbar.title.text == None


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
    assert converted.name == original.label


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
    assert converted.legendgroup == original.label
    assert converted.showlegend == first_trace


def test_two_lines(two_lines, Assert, not_core):
    graph = Graph(two_lines)
    fig = graph.to_plotly()
    assert len(fig.data) == 2
    first_trace = True
    for converted, y in zip(fig.data, two_lines.y):
        original = Series(x=two_lines.x, y=y, label=two_lines.label)
        compare_series(converted, original, Assert)
        check_legend_group(converted, original, first_trace)
        if first_trace:
            color = converted.line.color
            assert color is not None
            hex_color_regex = "^#(?:[0-9a-fA-F]{3}){1,2}$"
            assert re.match(hex_color_regex, color), f"'{color}' is not a hex color"
        assert converted.line.color == color
        first_trace = False


def compare_series_with_weight(converted, original, Assert):
    upper = original.y + original.weight
    lower = original.y - original.weight
    expected = Series(
        x=np.concatenate((original.x, original.x[::-1])),
        y=np.concatenate((lower, upper[::-1])),
        label=original.label,
    )
    compare_series(converted, expected, Assert)
    assert converted.mode == "none"
    assert converted.fill == "toself"
    assert converted.fillcolor is not None
    assert converted.opacity == 0.5


def test_fatband(fatband, Assert, not_core):
    graph = Graph(fatband)
    fig = graph.to_plotly()
    compare_series_with_weight(fig.data[0], fatband, Assert)


def test_two_fatbands(two_fatbands, Assert, not_core):
    graph = Graph(two_fatbands)
    fig = graph.to_plotly()
    assert len(fig.data) == 2
    first_trace = True
    for converted, y, w in zip(fig.data, two_fatbands.y, two_fatbands.weight):
        original = Series(x=two_fatbands.x, y=y, weight=w, label=two_fatbands.label)
        compare_series_with_weight(converted, original, Assert)
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
    Assert.allclose(fig.data[0].marker.size, fatband.weight)


def test_two_fatband_with_marker(two_fatbands, Assert, not_core):
    with_marker = dataclasses.replace(two_fatbands, marker="o")
    graph = Graph(with_marker)
    fig = graph.to_plotly()
    assert fig.layout.legend.itemsizing == "constant"
    assert len(fig.data) == 2
    for converted, y, w in zip(fig.data, two_fatbands.y, two_fatbands.weight):
        original = Series(x=two_fatbands.x, y=y, label=two_fatbands.label)
        compare_series(converted, original, Assert)
        assert converted.mode == "markers"
        Assert.allclose(converted.marker.size, w)


def test_weight_mode_color(two_fatbands, Assert, not_core):
    fatbands = dataclasses.replace(two_fatbands, marker="o", weight_mode="color")
    graph = Graph(fatbands)
    fig = graph.to_plotly()
    assert fig.layout.legend.itemsizing == "constant"
    assert len(fig.data) == 2
    for converted, y, w in zip(fig.data, two_fatbands.y, two_fatbands.weight):
        original = Series(x=two_fatbands.x, y=y, label=two_fatbands.label)
        compare_series(converted, original, Assert)
        assert converted.mode == "markers"
        Assert.allclose(converted.marker.color, w)
        assert converted.marker.coloraxis == "coloraxis"


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
    skipped_fields = ("series", "xsize", "ysize")
    init_all_fields = {
        field.name: field.name
        for field in dataclasses.fields(Graph)
        if field.name not in skipped_fields
    }
    graph1 = Graph(sine, **init_all_fields)
    graph2 = Graph(parabola)
    for field in dataclasses.fields(Graph):
        if field.name in skipped_fields:
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
    assert graph.series[0].label == "new label"
    assert parabola.label == "parabola"


def test_add_label_to_multiple_lines(parabola, sine, Assert):
    graph = Graph([parabola, sine]).label("new label")
    assert len(graph.series) == 2
    Assert.allclose(graph.series[0].x, parabola.x)
    Assert.allclose(graph.series[0].y, parabola.y)
    assert graph.series[0].label == "new label parabola"
    Assert.allclose(graph.series[1].x, sine.x)
    Assert.allclose(graph.series[1].y, sine.y)
    assert graph.series[1].label == "new label sine"


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
    Assert.allclose(df["two_fatbands.weight0"], two_fatbands.weight[0])
    Assert.allclose(df["two_fatbands.weight1"], two_fatbands.weight[1])


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
    pad_weight = len(parabola.x) - len(two_lines.x)
    pad_nan = np.repeat(np.nan, pad_weight)
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


def test_normal_series_does_not_set_contour_layout(parabola, not_core):
    graph = Graph(parabola)
    fig = graph.to_plotly()
    assert fig.layout.xaxis.visible is None
    assert fig.layout.yaxis.visible is None
    assert fig.layout.yaxis.scaleanchor is None


def test_contour(rectangle_contour, Assert, not_core):
    graph = Graph(rectangle_contour)
    fig = graph.to_plotly()
    assert len(fig.data) == 1
    # plotly expects y-x order
    Assert.allclose(fig.data[0].z, rectangle_contour.data.T)
    # shift because the points define the centers of the rectangles
    Assert.allclose(fig.data[0].x, np.linspace(0, 4, 20, endpoint=False) + 0.1)
    Assert.allclose(fig.data[0].y, np.linspace(0, 3.6, 18, endpoint=False) + 0.1)
    assert fig.data[0].name == rectangle_contour.label
    assert fig.data[0].autocontour
    # text explicitly that it is False to prevent None passing the test
    assert fig.layout.xaxis.visible == False
    assert fig.layout.yaxis.visible == False
    assert len(fig.layout.shapes) == 1
    check_unit_cell(fig.layout.shapes[0], x="4.0", y="3.6", zero="0.0")
    check_annotations(rectangle_contour.lattice, fig.layout.annotations, Assert)
    assert fig.layout.yaxis.scaleanchor == "x"


def test_contour_with_periodic_traces(rectangle_contour, Assert, not_core):
    rectangle_contour.traces_as_periodic = True
    graph = Graph(rectangle_contour)
    fig = graph.to_plotly()
    assert len(fig.data) == 1
    # plotly expects y-x order
    xdim, ydim = rectangle_contour.data.T.shape
    x_indices = np.arange(0, xdim + 1) % xdim
    y_indices = np.arange(0, ydim + 1) % ydim
    expected_data = rectangle_contour.data.T[np.ix_(x_indices, y_indices)]
    Assert.allclose(fig.data[0].z, expected_data)
    # shift because the points define the centers of the rectangles
    Assert.allclose(fig.data[0].x, np.linspace(0, 4, 21, endpoint=True))
    Assert.allclose(fig.data[0].y, np.linspace(0, 3.6, 19, endpoint=True))
    check_unit_cell(fig.layout.shapes[0], x="4.0", y="3.6", zero="0.0")
    check_annotations(rectangle_contour.lattice, fig.layout.annotations, Assert)


def test_contour_supercell(rectangle_contour, Assert, not_core):
    supercell = np.asarray((3, 5))
    rectangle_contour.supercell = supercell
    graph = Graph(rectangle_contour)
    fig = graph.to_plotly()
    assert len(fig.data) == 1
    # plotly expects y-x order
    assert all(fig.data[0].z.T.shape == supercell * rectangle_contour.data.shape)
    assert len(fig.data[0].x) == 60
    assert len(fig.data[0].y) == 90
    assert len(fig.layout.shapes) == 1
    check_unit_cell(fig.layout.shapes[0], x="4.0", y="3.6", zero="0.0")
    check_annotations(rectangle_contour.lattice, fig.layout.annotations, Assert)


def test_contour_supercell_with_periodic_traces(rectangle_contour, Assert, not_core):
    supercell = np.asarray((3, 5))
    rectangle_contour.traces_as_periodic = True
    rectangle_contour.supercell = supercell
    graph = Graph(rectangle_contour)
    fig = graph.to_plotly()
    assert len(fig.data) == 1
    # plotly expects y-x order
    assert all(
        fig.data[0].z.T.shape
        == np.array([ls + 1 for ls in supercell * rectangle_contour.data.shape])
    )
    assert len(fig.data[0].x) == 61
    assert len(fig.data[0].y) == 91
    assert len(fig.layout.shapes) == 1
    check_unit_cell(fig.layout.shapes[0], x="4.0", y="3.6", zero="0.0")
    check_annotations(rectangle_contour.lattice, fig.layout.annotations, Assert)


@pytest.mark.parametrize("show_contour_values", [True, False, None])
def test_contour_show_contour_values(rectangle_contour, show_contour_values, not_core):
    rectangle_contour.show_contour_values = show_contour_values
    graph = Graph(rectangle_contour)
    fig = graph.to_plotly()
    assert len(fig.data) == 1
    assert fig.data[0].contours.showlabels == show_contour_values


def test_legend_positioning(parabola, sine, Assert, not_core):
    graph = Graph([sine, parabola])
    fig = graph.to_plotly()
    assert len(fig.data) == 2
    assert (
        fig.layout.legend.x is None
    )  # actually at 1.02, but plotly does not expose this until set explicitly


def test_contour_legend_with_colorbar(rectangle_contour, parabola, Assert, not_core):
    graph = Graph([rectangle_contour, parabola])
    fig = graph.to_plotly()
    assert len(fig.data) == 2
    assert hasattr(fig.data[0], "colorbar") and fig.data[0].colorbar
    assert hasattr(fig.data[0].colorbar, "x")
    assert fig.data[0].colorbar.x == 1.02
    assert fig.layout.legend.x == pytest.approx(fig.data[0].colorbar.x + 0.18, abs=1e-6)


def check_unit_cell(unit_cell, x, y, zero):
    assert unit_cell.type == "path"
    assert unit_cell.path == f"M 0 0 L {x} {zero} L {x} {y} L {zero} {y} Z"
    assert unit_cell.line.color == _config.VASP_COLORS["dark"]


def check_annotations(lattice, annotations, Assert):
    assert len(lattice.vectors) == len(annotations)
    sign = np.sign(np.cross(*lattice.vectors))
    labels = "abc".replace(lattice.cut, "")
    for vector, label, annotation in zip(lattice.vectors, labels, annotations):
        assert annotation.showarrow == False
        assert annotation.text == label
        Assert.allclose((annotation.x, annotation.y), 0.5 * vector)
        expected_shift = sign * 10 * vector[::-1] / np.linalg.norm(vector)
        Assert.allclose((annotation.xshift, -annotation.yshift), expected_shift)
        sign *= -1


def test_contour_interpolate(tilted_contour, Assert, not_core):
    tilted_contour.color_scheme = "auto"
    graph = Graph(tilted_contour)
    fig = graph.to_plotly()
    area_cell = 12.0
    points_per_area = tilted_contour.data.size / area_cell
    points_per_line = np.sqrt(points_per_area) * tilted_contour._interpolation_factor
    lengths = np.array([6, 9])  # this accounts for the 2 x 1 supercell
    expected_shape = np.ceil(points_per_line * lengths).astype(int)
    expected_average = np.average(tilted_contour.data)
    assert len(fig.data) == 1
    # plotly expects y-x order
    assert all(fig.data[0].z.T.shape == expected_shape)
    assert fig.data[0].x.size == expected_shape[0]
    assert fig.data[0].y.size == expected_shape[1]
    finite = np.isfinite(fig.data[0].z)
    assert np.isclose(np.average(fig.data[0].z[finite]), expected_average)
    assert np.any(finite)
    assert len(fig.layout.shapes) == 0
    # This colorscheme will be diverging due to interpolation
    expected_colorscale = px.colors.sequential.RdBu_r
    assert len(fig.data[0].colorscale) == len(expected_colorscale)
    for actual, expected in zip(fig.data[0].colorscale, expected_colorscale):
        assert actual[1] == expected
    check_annotations(tilted_contour.lattice, fig.layout.annotations, Assert)


# @pytest.mark.xfail
def test_contour_interpolate_with_periodic_traces(tilted_contour, Assert, not_core):
    tilted_contour.traces_as_periodic = True
    tilted_contour.color_scheme = "auto"
    graph = Graph(tilted_contour)
    fig = graph.to_plotly()

    xdim, ydim = tilted_contour.data.shape
    x_indices = np.arange(0, xdim + 1) % xdim
    y_indices = np.arange(0, ydim + 1) % ydim
    expected_data = tilted_contour.data[np.ix_(x_indices, y_indices)]
    expected_average = np.average(expected_data.data)
    assert len(fig.data) == 1
    # plotly expects y-x order
    finite = np.isfinite(fig.data[0].z)
    assert np.any(finite)
    assert np.isclose(np.average(fig.data[0].z[finite]), expected_average, rtol=0.1)
    assert len(fig.layout.shapes) == 0
    expected_colorscale = px.colors.sequential.Reds
    assert len(fig.data[0].colorscale) == len(expected_colorscale)
    for actual, expected in zip(fig.data[0].colorscale, expected_colorscale):
        assert actual[1] == expected
    check_annotations(tilted_contour.lattice, fig.layout.annotations, Assert)


def test_mix_contour_and_series(two_lines, rectangle_contour, not_core):
    graph = Graph([rectangle_contour, two_lines])
    fig = graph.to_plotly()
    assert len(fig.data) == 3
    assert fig.layout.xaxis.visible is None
    assert fig.layout.yaxis.visible is None
    assert fig.layout.yaxis.scaleanchor == "x"


def test_simple_quiver(simple_quiver, Assert, not_core):
    graph = Graph(simple_quiver)
    assert simple_quiver.max_number_arrows is None
    expected_positions, dx, dy = compute_positions(simple_quiver)
    work = simple_quiver.scale_arrows * simple_quiver.data.T.reshape(-1, 2)
    expected_positions -= 0.5 * work
    expected_positions = expected_positions + 0.5 * np.array([dx, dy])
    expected_tips = expected_positions + work
    expected_barb_length = 0.3 * np.linalg.norm(work, axis=-1).flatten()
    #
    fig = graph.to_plotly()
    data_size = simple_quiver.data.size // 2
    assert len(fig.data) == 1
    actual = split_data(fig.data[0], data_size, Assert)
    assert len(fig.layout.shapes) == 1
    check_unit_cell(fig.layout.shapes[0], x="3", y="5", zero="0")
    check_annotations(simple_quiver.lattice, fig.layout.annotations, Assert)
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.data[0].line.color == _config.VASP_COLORS["dark"]
    Assert.allclose(actual.positions, expected_positions)
    Assert.allclose(actual.tips, expected_tips)
    Assert.allclose(actual.barb_length, expected_barb_length)


def test_simple_quiver_with_periodic_traces(simple_quiver, Assert, not_core):
    simple_quiver.traces_as_periodic = True
    graph = Graph(simple_quiver)
    assert simple_quiver.max_number_arrows is None
    expected_positions, _, _ = compute_positions(simple_quiver, periodic=True)
    xdim, ydim, _ = simple_quiver.data.T.shape
    x_indices = np.arange(0, xdim + 1) % xdim
    y_indices = np.arange(0, ydim + 1) % ydim
    expected_data = simple_quiver.data.T[np.ix_(x_indices, y_indices)]
    work = simple_quiver.scale_arrows * expected_data.reshape(-1, 2)
    assert work.shape == expected_positions.shape
    expected_positions -= 0.5 * work
    expected_tips = expected_positions + work
    expected_barb_length = 0.3 * np.linalg.norm(work, axis=-1).flatten()
    #
    fig = graph.to_plotly()
    data_size = expected_data.size // 2
    assert len(fig.data) == 1
    actual = split_data(fig.data[0], data_size, Assert)
    assert len(fig.layout.shapes) == 1
    check_unit_cell(fig.layout.shapes[0], x="3", y="5", zero="0")
    check_annotations(simple_quiver.lattice, fig.layout.annotations, Assert)
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.data[0].line.color == _config.VASP_COLORS["dark"]
    Assert.allclose(actual.positions, expected_positions)
    Assert.allclose(actual.tips, expected_tips)
    Assert.allclose(actual.barb_length, expected_barb_length)


@pytest.mark.parametrize("max_number_arrows", (1025, 1024, 680))
def test_dense_quiver(dense_quiver, max_number_arrows, Assert, not_core):
    dense_quiver.max_number_arrows = max_number_arrows
    if max_number_arrows == 1024:
        expected_shape = (28, 25)
        subsampling = (3, 3)
    elif max_number_arrows == 1025:
        expected_shape = (41, 25)
        subsampling = (2, 3)
    elif max_number_arrows == 680:
        expected_shape = (28, 19)
        subsampling = (3, 4)
    else:
        raise NotImplemented
    graph = Graph(dense_quiver)
    work = dense_quiver.scale_arrows * dense_quiver.data
    work = np.block([[work, work, work], [work, work, work]]).T
    # remember that a and b are transposed
    work = work[:: subsampling[1], :: subsampling[0]]
    expected_positions, dx, dy = compute_positions(dense_quiver, subsampling)
    expected_positions -= 0.5 * work.reshape(expected_positions.shape)
    expected_positions = expected_positions + 0.5 * np.array([dx, dy])
    expected_tips = expected_positions + work.reshape(expected_positions.shape)
    expected_barb_length = 0.3 * np.linalg.norm(work, axis=-1).flatten()
    data_size = np.prod(expected_shape)
    #
    fig = graph.to_plotly()
    assert len(fig.data) == 1
    actual = split_data(fig.data[0], data_size, Assert)
    Assert.allclose(actual.positions, expected_positions)
    Assert.allclose(actual.tips, expected_tips)
    Assert.allclose(actual.barb_length, expected_barb_length)


@pytest.mark.parametrize("max_number_arrows", (1025, 1024, 680))
def test_dense_quiver_with_periodic_traces(
    dense_quiver, max_number_arrows, Assert, not_core
):
    dense_quiver.traces_as_periodic = True
    dense_quiver.max_number_arrows = max_number_arrows
    if max_number_arrows == 1024:
        expected_shape = (28, 26)
        subsampling = (3, 3)
    elif max_number_arrows == 1025:
        expected_shape = (42, 26)
        subsampling = (2, 3)
    elif max_number_arrows == 680:
        expected_shape = (28, 19)
        subsampling = (3, 4)
    else:
        raise NotImplemented
    graph = Graph(dense_quiver)
    work = dense_quiver.scale_arrows * dense_quiver.data
    work = np.tile(work, (1, dense_quiver.supercell[0], dense_quiver.supercell[1]))
    # expand work
    zdim, ydim, xdim = work.shape
    z_indices = np.arange(0, zdim)
    x_indices = np.arange(0, xdim + 1) % xdim
    y_indices = np.arange(0, ydim + 1) % ydim
    work = work[np.ix_(z_indices, y_indices, x_indices)]
    work = np.transpose(work, (2, 1, 0))
    # remember that a and b are transposed
    work = work[:: subsampling[1], :: subsampling[0]]
    expected_positions, dx, dy = compute_positions(
        dense_quiver, subsampling, periodic=True
    )
    expected_positions -= 0.5 * work.reshape(expected_positions.shape)
    expected_tips = expected_positions + work.reshape(expected_positions.shape)
    expected_barb_length = 0.3 * np.linalg.norm(work, axis=-1).flatten()
    data_size = np.prod(expected_shape)
    #
    fig = graph.to_plotly()
    assert len(fig.data) == 1
    actual = split_data(fig.data[0], data_size, Assert)
    Assert.allclose(actual.positions, expected_positions)
    Assert.allclose(actual.tips, expected_tips)
    Assert.allclose(actual.barb_length, expected_barb_length)


def test_complex_quiver(complex_quiver, Assert, not_core):
    graph = Graph(complex_quiver)
    expected_scale = 0.10015332542313245
    work = expected_scale * complex_quiver.data
    work = np.block([[work, work], [work, work], [work, work]]).T
    expected_positions, dx, dy = compute_positions(complex_quiver)
    expected_positions -= 0.5 * work.reshape(expected_positions.shape)
    expected_positions = expected_positions + 0.5 * np.array([dx, dy])
    expected_tips = expected_positions + work.reshape(expected_positions.shape)
    expected_barb_length = 0.3 * np.linalg.norm(work, axis=-1).flatten()
    data_size = np.prod(complex_quiver.supercell) * complex_quiver.data.size // 2
    #
    fig = graph.to_plotly()
    assert len(fig.data) == 1
    actual = split_data(fig.data[0], data_size, Assert)
    Assert.allclose(actual.positions, expected_positions)
    Assert.allclose(actual.tips, expected_tips)
    Assert.allclose(actual.barb_length, expected_barb_length)
    assert len(fig.layout.annotations) == 0


def test_complex_quiver_with_periodic_traces(complex_quiver, Assert, not_core):
    complex_quiver.traces_as_periodic = True
    graph = Graph(complex_quiver)
    expected_scale = 0.10015332542313245
    work = expected_scale * complex_quiver.data
    work = np.tile(work, (1, complex_quiver.supercell[0], complex_quiver.supercell[1]))
    # expand work
    zdim, ydim, xdim = work.shape
    z_indices = np.arange(0, zdim)
    x_indices = np.arange(0, xdim + 1) % xdim
    y_indices = np.arange(0, ydim + 1) % ydim
    work = work[np.ix_(z_indices, y_indices, x_indices)]
    work = np.transpose(work, (2, 1, 0))
    expected_positions, dx, dy = compute_positions(complex_quiver, periodic=True)
    work = work.reshape(expected_positions.shape)
    assert work.shape == expected_positions.shape
    expected_positions -= 0.5 * work
    expected_tips = expected_positions + work
    expected_barb_length = 0.3 * np.linalg.norm(work, axis=-1).flatten()
    data_size = work.shape[0]
    #
    fig = graph.to_plotly()
    assert len(fig.data) == 1
    actual = split_data(fig.data[0], data_size, Assert)
    Assert.allclose(actual.positions, expected_positions)
    Assert.allclose(actual.tips, expected_tips)
    Assert.allclose(actual.barb_length, expected_barb_length)
    assert len(fig.layout.annotations) == 0


def test_width_and_height(parabola, not_core):
    fig = Graph(parabola).to_plotly()
    assert fig.layout.width == 720
    assert fig.layout.height == 540
    fig = Graph(parabola, xsize=800, ysize=600).to_plotly()
    assert fig.layout.width == 800
    assert fig.layout.height == 600


def test_range_for_x_and_y_axis(parabola, Assert, not_core):
    fig = Graph(parabola).to_plotly()
    assert fig.layout.xaxis.range is None
    assert fig.layout.yaxis.range is None
    graph = Graph(parabola, xrange=[-1.5, 7.2], yrange=(2.1, 4.3))
    fig = graph.to_plotly()
    Assert.allclose(fig.layout.xaxis.range, graph.xrange)
    Assert.allclose(fig.layout.yaxis.range, graph.yrange)


@dataclasses.dataclass
class ContourData:
    positions: np.ndarray = None
    tips: np.ndarray = None
    barb_length: np.ndarray = None


def compute_positions(contour, subsampling=(1, 1), periodic=False):
    step_a = np.divide(contour.lattice.vectors[0], contour.data.shape[1])
    step_b = np.divide(contour.lattice.vectors[1], contour.data.shape[2])
    shape = np.multiply(contour.supercell, contour.data.shape[1:])
    # remember that the data is transposed
    range_a = range(0, shape[0] + (1 if periodic else 0), subsampling[0])
    range_b = range(0, shape[1] + (1 if periodic else 0), subsampling[1])
    dx = np.linalg.norm(step_a)
    dy = np.linalg.norm(step_b)
    return np.array([a * step_a + b * step_b for b in range_b for a in range_a]), dx, dy


def split_data(data, data_size, Assert):
    actual = ContourData()
    assert data.mode == "lines"
    # The data contains first a line between each grid point and the tip of the arrow;
    # each of the lines are separated by a None. The second part is the tip of the arrow
    # consisting of two lines; again tips are separated by None.
    assert len(data.x) == len(data.y) == 7 * data_size
    # first element contains positions
    slice_ = slice(0, 3 * data_size, 3)
    actual.positions = np.array((data.x[slice_], data.y[slice_])).T
    # second element of both parts contain the tip of the arrows
    slice_ = slice(1, 3 * data_size, 3)
    actual.tips = np.array((data.x[slice_], data.y[slice_])).T
    slice_ = slice(3 * data_size + 1, None, 4)
    other_tips = np.array((data.x[slice_], data.y[slice_])).T
    Assert.allclose(other_tips, actual.tips)
    # third element of first part and fourth element of second part should be None (=separator)
    slice_ = slice(2, 3 * data_size, 3)
    assert all(element is None for element in data.x[slice_])
    assert all(element is None for element in data.y[slice_])
    slice_ = slice(3 * data_size + 3, None, 4)
    assert all(element is None for element in data.x[slice_])
    assert all(element is None for element in data.y[slice_])
    # the first and third element of the second part contain the barb of the arrow
    slice_ = slice(3 * data_size, None, 4)
    barb = np.array((data.x[slice_], data.y[slice_])).T
    actual.barb_length = np.linalg.norm(barb - actual.tips, axis=-1)
    slice_ = slice(3 * data_size + 2, None, 4)
    other_barb = np.array((data.x[slice_], data.y[slice_])).T
    other_barb_length = np.linalg.norm(other_barb - actual.tips, axis=-1)
    Assert.allclose(other_barb_length, actual.barb_length, tolerance=10)
    return actual


def test_no_common_names():
    assert set(Graph._fields).intersection(Series._fields) == set()
