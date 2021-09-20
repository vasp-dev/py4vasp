from py4vasp.data import Dielectric
import dataclasses
import numpy as np
import pytest
import types
from unittest.mock import patch


@pytest.fixture
def dielectric(raw_data):
    raw_dielectric = raw_data.dielectric("default")
    dielectric = Dielectric(raw_dielectric)
    dielectric.ref = types.SimpleNamespace()
    dielectric.ref.energies = raw_dielectric.energies
    epsilon = raw_dielectric.function[..., 0] + 1j * raw_dielectric.function[..., 1]
    dielectric.ref.function = epsilon
    dielectric.ref.isotropic = np.trace(epsilon) / 3
    dielectric.ref.xx = epsilon[0, 0]
    dielectric.ref.yy = epsilon[1, 1]
    dielectric.ref.zz = epsilon[2, 2]
    dielectric.ref.xy = 0.5 * (epsilon[0, 1] + epsilon[1, 0])
    dielectric.ref.yz = 0.5 * (epsilon[1, 2] + epsilon[2, 1])
    dielectric.ref.xz = 0.5 * (epsilon[0, 2] + epsilon[2, 0])
    return dielectric


def test_dielectric_read(dielectric, Assert):
    actual = dielectric.read()
    Assert.allclose(actual["energies"], dielectric.ref.energies)
    Assert.allclose(actual["function"], dielectric.ref.function)


@dataclasses.dataclass
class Plot:
    x: np.ndarray
    y: np.ndarray
    name: str


def test_dielectric_plot_default(dielectric, Assert):
    plots = [
        Plot(
            x=dielectric.ref.energies,
            y=dielectric.ref.isotropic.real,
            name=expected_plot_name("Re", "isotropic"),
        ),
        Plot(
            x=dielectric.ref.energies,
            y=dielectric.ref.isotropic.imag,
            name=expected_plot_name("Im", "isotropic"),
        ),
    ]
    fig = dielectric.plot()
    check_figure_contains_plots(fig, plots, Assert)


def test_dielectric_plot_direction(dielectric, Assert):
    directions = ("xx", "yy", "zz", "xy", "yz", "xz")
    for direction in directions:
        plots = [
            Plot(
                x=dielectric.ref.energies,
                y=getattr(dielectric.ref, direction).real,
                name=expected_plot_name("Re", direction),
            ),
            Plot(
                x=dielectric.ref.energies,
                y=getattr(dielectric.ref, direction).imag,
                name=expected_plot_name("Im", direction),
            ),
        ]
        fig = dielectric.plot(direction)
        check_figure_contains_plots(fig, plots, Assert)


def test_dielectric_plot_component(dielectric, Assert):
    real_plot = Plot(
        x=dielectric.ref.energies,
        y=dielectric.ref.isotropic.real,
        name=expected_plot_name("Re", "isotropic"),
    )
    imag_plot = Plot(
        x=dielectric.ref.energies,
        y=dielectric.ref.isotropic.imag,
        name=expected_plot_name("Im", "isotropic"),
    )
    for component in ("real", "Re"):
        fig = dielectric.plot(component)
        check_figure_contains_plots(fig, [real_plot], Assert)
    for component in ("imaginary", "imag", "Im"):
        fig = dielectric.plot(component)
        check_figure_contains_plots(fig, [imag_plot], Assert)


def test_dielectric_plot_nested(dielectric, Assert):
    plots = [
        Plot(
            x=dielectric.ref.energies,
            y=dielectric.ref.xx.real,
            name=expected_plot_name("Re", "xx"),
        ),
        Plot(
            x=dielectric.ref.energies,
            y=dielectric.ref.xy.imag,
            name=expected_plot_name("Im", "xy"),
        ),
        Plot(
            x=dielectric.ref.energies,
            y=dielectric.ref.yz.imag,
            name=expected_plot_name("Im", "yz"),
        ),
        Plot(
            x=dielectric.ref.energies,
            y=dielectric.ref.zz.real,
            name=expected_plot_name("Re", "zz"),
        ),
        Plot(
            x=dielectric.ref.energies,
            y=dielectric.ref.zz.imag,
            name=expected_plot_name("Im", "zz"),
        ),
    ]
    fig = dielectric.plot("Re(xx) Im(xy, yz) zz(Re, Im)")
    check_figure_contains_plots(fig, plots, Assert)


def expected_plot_name(component, direction):
    subscript = "" if direction == "isotropic" else f"_{{{direction}}}"
    return f"{component}($\\epsilon{subscript}$)"


def check_figure_contains_plots(fig, references, Assert):
    assert fig.layout.xaxis.title.text == "Energy (eV)"
    assert fig.layout.yaxis.title.text == r"$\epsilon$"
    assert len(fig.data) == len(references)
    for data, ref in zip(fig.data, references):
        Assert.allclose(data.x, ref.x)
        Assert.allclose(data.y, ref.y)
        assert data.name == ref.name


def test_dielectric_to_image(dielectric):
    check_to_image(dielectric, None, "dielectric.png")
    custom_filename = "custom.jpg"
    check_to_image(dielectric, custom_filename, custom_filename)


def check_to_image(dielectric, filename_argument, expected_filename):
    with patch("py4vasp.data.dielectric._to_plotly") as plot:
        dielectric.to_image("args", filename=filename_argument, key="word")
        plot.assert_called_once()
        assert plot.call_args.args[1] == "args"
        assert plot.call_args.kwargs == {"key": "word"}
        fig = plot.return_value
        fig.write_image.assert_called_once_with(dielectric._path / expected_filename)


def test_dielectric_print(dielectric, format_):
    actual, _ = format_(dielectric)
    reference = f"""
dielectric function:
    energies: [0.00, 1.00] 50 points
    directions: isotropic, xx, yy, zz, xy, yz, xz
    """.strip()
    assert actual == {"text/plain": reference}


def test_descriptor(dielectric, check_descriptors):
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_to_plotly": ["to_plotly", "plot"],
    }
    check_descriptors(dielectric, descriptors)


def test_from_file(raw_data, mock_file, check_read):
    raw_dielectric = raw_data.dielectric("default")
    with mock_file("dielectric", raw_dielectric) as mocks:
        check_read(Dielectric, mocks, raw_dielectric)
