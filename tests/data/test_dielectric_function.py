# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import types
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception
from py4vasp.data import DielectricFunction


@pytest.fixture
def electronic(raw_data):
    raw_electronic = raw_data.dielectric_function("electron")
    electronic = DielectricFunction.from_data(raw_electronic)
    electronic.ref = types.SimpleNamespace()
    electronic.ref.energies = raw_electronic.energies
    to_complex = lambda data: data[..., 0] + 1j * data[..., 1]
    electronic.ref.dielectric_function = to_complex(raw_electronic.dielectric_function)
    electronic.ref.current_current = to_complex(raw_electronic.current_current)
    return electronic


@pytest.fixture
def ionic(raw_data):
    raw_ionic = raw_data.dielectric_function("ion")
    ionic = DielectricFunction.from_data(raw_ionic)
    ionic.ref = types.SimpleNamespace()
    ionic.ref.energies = raw_ionic.energies
    to_complex = lambda data: data[..., 0] + 1j * data[..., 1]
    ionic.ref.dielectric_function = to_complex(raw_ionic.dielectric_function)
    return ionic


def test_electronic_read(electronic, Assert):
    check_dielectric_read(electronic, Assert)


def test_ionic_read(ionic, Assert):
    check_dielectric_read(ionic, Assert)


def check_dielectric_read(dielectric_function, Assert):
    for method in (dielectric_function.read, dielectric_function.to_dict):
        actual = method()
        reference = dielectric_function.ref
        Assert.allclose(actual["energies"], reference.energies)
        Assert.allclose(actual["dielectric_function"], reference.dielectric_function)
        if hasattr(reference, "current_current"):
            Assert.allclose(actual["current_current"], reference.current_current)
        else:
            assert "current_current" not in actual


@dataclasses.dataclass
class Plot:
    x: np.ndarray
    y: np.ndarray
    name: str


def test_electronic_plot_default(electronic, Assert):
    plots = [
        Plot(
            x=electronic.ref.energies,
            y=isotropic(electronic.ref.dielectric_function).real,
            name=expected_plot_name("Re", "isotropic", "density"),
        ),
        Plot(
            x=electronic.ref.energies,
            y=isotropic(electronic.ref.dielectric_function).imag,
            name=expected_plot_name("Im", "isotropic", "density"),
        ),
    ]
    fig = electronic.plot()
    check_figure_contains_plots(fig, plots, Assert)


def test_ionic_plot_default(ionic, Assert):
    plots = [
        Plot(
            x=ionic.ref.energies,
            y=isotropic(ionic.ref.dielectric_function).real,
            name=expected_plot_name("Re", "isotropic"),
        ),
        Plot(
            x=ionic.ref.energies,
            y=isotropic(ionic.ref.dielectric_function).imag,
            name=expected_plot_name("Im", "isotropic"),
        ),
    ]
    fig = ionic.plot()
    check_figure_contains_plots(fig, plots, Assert)


def test_electronic_plot_component(electronic, Assert):
    density_plots = [
        Plot(
            x=electronic.ref.energies,
            y=isotropic(electronic.ref.dielectric_function).real,
            name=expected_plot_name("Re", "isotropic", "density"),
        ),
        Plot(
            x=electronic.ref.energies,
            y=isotropic(electronic.ref.dielectric_function).imag,
            name=expected_plot_name("Im", "isotropic", "density"),
        ),
    ]
    current_plots = [
        Plot(
            x=electronic.ref.energies,
            y=isotropic(electronic.ref.current_current).real,
            name=expected_plot_name("Re", "isotropic", "current"),
        ),
        Plot(
            x=electronic.ref.energies,
            y=isotropic(electronic.ref.current_current).imag,
            name=expected_plot_name("Im", "isotropic", "current"),
        ),
    ]
    fig = electronic.plot("density")
    check_figure_contains_plots(fig, density_plots, Assert)
    fig = electronic.plot("current")
    check_figure_contains_plots(fig, current_plots, Assert)


def test_electronic_plot_direction(electronic, Assert):
    directions = ("xx", "yy", "zz", "xy", "yz", "xz")
    for direction in directions:
        reference = get_direction(electronic.ref.dielectric_function, direction)
        plots = [
            Plot(
                x=electronic.ref.energies,
                y=reference.real,
                name=expected_plot_name("Re", direction, "density"),
            ),
            Plot(
                x=electronic.ref.energies,
                y=reference.imag,
                name=expected_plot_name("Im", direction, "density"),
            ),
        ]
        fig = electronic.plot(direction)
        check_figure_contains_plots(fig, plots, Assert)


def test_ionic_plot_direction(ionic, Assert):
    directions = ("xx", "yy", "zz", "xy", "yz", "xz")
    for direction in directions:
        reference = get_direction(ionic.ref.dielectric_function, direction)
        plots = [
            Plot(
                x=ionic.ref.energies,
                y=reference.real,
                name=expected_plot_name("Re", direction),
            ),
            Plot(
                x=ionic.ref.energies,
                y=reference.imag,
                name=expected_plot_name("Im", direction),
            ),
        ]
        fig = ionic.plot(direction)
        check_figure_contains_plots(fig, plots, Assert)


def test_electronic_plot_real_or_imag(electronic, Assert):
    real_plot = Plot(
        x=electronic.ref.energies,
        y=isotropic(electronic.ref.dielectric_function).real,
        name=expected_plot_name("Re", "isotropic", "density"),
    )
    imag_plot = Plot(
        x=electronic.ref.energies,
        y=isotropic(electronic.ref.dielectric_function).imag,
        name=expected_plot_name("Im", "isotropic", "density"),
    )
    for real in ("real", "Re"):
        fig = electronic.plot(real)
        check_figure_contains_plots(fig, [real_plot], Assert)
    for imag in ("imaginary", "imag", "Im"):
        fig = electronic.plot(imag)
        check_figure_contains_plots(fig, [imag_plot], Assert)


def test_ionic_plot_real_or_imag(ionic, Assert):
    real_plot = Plot(
        x=ionic.ref.energies,
        y=isotropic(ionic.ref.dielectric_function).real,
        name=expected_plot_name("Re", "isotropic"),
    )
    imag_plot = Plot(
        x=ionic.ref.energies,
        y=isotropic(ionic.ref.dielectric_function).imag,
        name=expected_plot_name("Im", "isotropic"),
    )
    for real in ("real", "Re"):
        fig = ionic.plot(real)
        check_figure_contains_plots(fig, [real_plot], Assert)
    for imag in ("imaginary", "imag", "Im"):
        fig = ionic.plot(imag)
        check_figure_contains_plots(fig, [imag_plot], Assert)


def test_electronic_plot_nested(electronic, Assert):
    plots = [
        Plot(
            x=electronic.ref.energies,
            y=get_direction(electronic.ref.dielectric_function, "xx").real,
            name=expected_plot_name("Re", "xx", "density"),
        ),
        Plot(
            x=electronic.ref.energies,
            y=get_direction(electronic.ref.current_current, "xy").imag,
            name=expected_plot_name("Im", "xy", "current"),
        ),
        Plot(
            x=electronic.ref.energies,
            y=get_direction(electronic.ref.current_current, "yz").imag,
            name=expected_plot_name("Im", "yz", "current"),
        ),
        Plot(
            x=electronic.ref.energies,
            y=isotropic(electronic.ref.dielectric_function).real,
            name=expected_plot_name("Re", "isotropic", "density"),
        ),
        Plot(
            x=electronic.ref.energies,
            y=isotropic(electronic.ref.current_current).real,
            name=expected_plot_name("Re", "isotropic", "current"),
        ),
    ]
    selection = "density(Re(xx)) Im(current(xy,yz)) Re(density,current)"
    fig = electronic.plot(selection)
    check_figure_contains_plots(fig, plots, Assert)


def test_ionic_plot_nested(ionic, Assert):
    plots = [
        Plot(
            x=ionic.ref.energies,
            y=get_direction(ionic.ref.dielectric_function, "xx").real,
            name=expected_plot_name("Re", "xx"),
        ),
        Plot(
            x=ionic.ref.energies,
            y=get_direction(ionic.ref.dielectric_function, "xx").imag,
            name=expected_plot_name("Im", "xx"),
        ),
        Plot(
            x=ionic.ref.energies,
            y=get_direction(ionic.ref.dielectric_function, "zz").real,
            name=expected_plot_name("Re", "zz"),
        ),
        Plot(
            x=ionic.ref.energies,
            y=get_direction(ionic.ref.dielectric_function, "zz").imag,
            name=expected_plot_name("Im", "zz"),
        ),
    ]
    selection = "xx zz(Re,Im)"
    fig = ionic.plot(selection)
    check_figure_contains_plots(fig, plots, Assert)


def test_incorrect_direction_raises_error(electronic):
    with pytest.raises(exception.IncorrectUsage):
        electronic.plot("incorrect")


def isotropic(tensor):
    return np.trace(tensor) / 3


def get_direction(tensor, direction):
    lookup = {"x": 0, "y": 1, "z": 2}
    i = lookup[direction[0]]
    j = lookup[direction[1]]
    return 0.5 * (tensor[i, j] + tensor[j, i])


def expected_plot_name(real_or_imag, direction, component=None):
    parts = (real_or_imag,)
    if component is not None:
        parts += (component,)
    if direction != "isotropic":
        parts += (direction,)
    return "_".join(parts)


def check_figure_contains_plots(fig, references, Assert):
    assert fig.xlabel == "Energy (eV)"
    assert fig.ylabel == "dielectric function ϵ"
    assert len(fig.series) == len(references)
    for data, ref in zip(fig.series, references):
        Assert.allclose(data.x, ref.x)
        Assert.allclose(data.y, ref.y)
        assert data.name == ref.name


@patch("py4vasp._data.dielectric_function.DielectricFunction.to_graph")
def test_electronic_to_plotly(mock_plot, electronic):
    fig = electronic.to_plotly("selection")
    mock_plot.assert_called_once_with("selection")
    graph = mock_plot.return_value
    graph.to_plotly.assert_called_once()
    assert fig == graph.to_plotly.return_value


def test_electronic_to_image(electronic):
    check_to_image(electronic, None, "dielectric_function.png")
    custom_filename = "custom.jpg"
    check_to_image(electronic, custom_filename, custom_filename)


def test_ionic_to_image(ionic):
    check_to_image(ionic, None, "dielectric_function.png")
    custom_filename = "custom.jpg"
    check_to_image(ionic, custom_filename, custom_filename)


def check_to_image(dielectric_function, filename_argument, expected_filename):
    plot_function = "py4vasp._data.dielectric_function.DielectricFunction.to_plotly"
    with patch(plot_function) as plot:
        dielectric_function.to_image("args", filename=filename_argument, key="word")
        plot.assert_called_once_with("args", key="word")
        fig = plot.return_value
        expected_path = dielectric_function.path / expected_filename
        fig.write_image.assert_called_once_with(expected_path)


def test_electronic_print(electronic, format_):
    actual, _ = format_(electronic)
    reference = f"""
dielectric function:
    energies: [0.00, 1.00] 50 points
    components: density, current
    directions: isotropic, xx, yy, zz, xy, yz, xz
    """.strip()
    assert actual == {"text/plain": reference}


def test_ionic_print(ionic, format_):
    actual, _ = format_(ionic)
    reference = f"""
dielectric function:
    energies: [0.00, 1.00] 50 points
    directions: isotropic, xx, yy, zz, xy, yz, xz
    """.strip()
    assert actual == {"text/plain": reference}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.dielectric_function("electron")
    check_factory_methods(DielectricFunction, data)
