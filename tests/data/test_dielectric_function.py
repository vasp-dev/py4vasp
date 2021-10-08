from py4vasp.data import DielectricFunction
import dataclasses
import numpy as np
import pytest
import types
from unittest.mock import patch


@pytest.fixture
def dielectric_function(raw_data):
    raw_dielectric = raw_data.dielectric_function("default")
    dielectric = DielectricFunction(raw_dielectric)
    dielectric.ref = types.SimpleNamespace()
    dielectric.ref.energies = raw_dielectric.energies
    to_complex = lambda data: data[..., 0] + 1j * data[..., 1]
    dielectric.ref.density_density = to_complex(raw_dielectric.density_density)
    dielectric.ref.current_current = to_complex(raw_dielectric.current_current)
    dielectric.ref.ion = to_complex(raw_dielectric.ion)
    dielectric.ref.isotropic = np.trace(dielectric.ref.density_density) / 3
    return dielectric


def test_dielectric_read(dielectric_function, Assert):
    actual = dielectric_function.read()
    Assert.allclose(actual["energies"], dielectric_function.ref.energies)
    Assert.allclose(actual["density_density"], dielectric_function.ref.density_density)
    Assert.allclose(actual["current_current"], dielectric_function.ref.current_current)
    Assert.allclose(actual["ion"], dielectric_function.ref.ion)


@dataclasses.dataclass
class Plot:
    x: np.ndarray
    y: np.ndarray
    name: str


def test_dielectric_plot_default(dielectric_function, Assert):
    plots = [
        Plot(
            x=dielectric_function.ref.energies,
            y=isotropic(dielectric_function.ref.density_density).real,
            name=expected_plot_name("density", "Re", "isotropic"),
        ),
        Plot(
            x=dielectric_function.ref.energies,
            y=isotropic(dielectric_function.ref.density_density).imag,
            name=expected_plot_name("density", "Im", "isotropic"),
        ),
    ]
    fig = dielectric_function.plot()
    check_figure_contains_plots(fig, plots, Assert)


def test_dielectric_plot_component(dielectric_function, Assert):
    density_plots = [
        Plot(
            x=dielectric_function.ref.energies,
            y=isotropic(dielectric_function.ref.density_density).real,
            name=expected_plot_name("density", "Re", "isotropic"),
        ),
        Plot(
            x=dielectric_function.ref.energies,
            y=isotropic(dielectric_function.ref.density_density).imag,
            name=expected_plot_name("density", "Im", "isotropic"),
        ),
    ]
    current_plots = [
        Plot(
            x=dielectric_function.ref.energies,
            y=isotropic(dielectric_function.ref.current_current).real,
            name=expected_plot_name("current", "Re", "isotropic"),
        ),
        Plot(
            x=dielectric_function.ref.energies,
            y=isotropic(dielectric_function.ref.current_current).imag,
            name=expected_plot_name("current", "Im", "isotropic"),
        ),
    ]
    ion_plots = [
        Plot(
            x=dielectric_function.ref.energies,
            y=isotropic(dielectric_function.ref.ion).real,
            name=expected_plot_name("ion", "Re", "isotropic"),
        ),
        Plot(
            x=dielectric_function.ref.energies,
            y=isotropic(dielectric_function.ref.ion).imag,
            name=expected_plot_name("ion", "Im", "isotropic"),
        ),
    ]
    fig = dielectric_function.plot("density")
    check_figure_contains_plots(fig, density_plots, Assert)
    fig = dielectric_function.plot("current")
    check_figure_contains_plots(fig, current_plots, Assert)
    fig = dielectric_function.plot("ion")
    check_figure_contains_plots(fig, ion_plots, Assert)


def test_dielectric_plot_direction(dielectric_function, Assert):
    directions = ("xx", "yy", "zz", "xy", "yz", "xz")
    for direction in directions:
        reference = get_direction(dielectric_function.ref.density_density, direction)
        plots = [
            Plot(
                x=dielectric_function.ref.energies,
                y=reference.real,
                name=expected_plot_name("density", "Re", direction),
            ),
            Plot(
                x=dielectric_function.ref.energies,
                y=reference.imag,
                name=expected_plot_name("density", "Im", direction),
            ),
        ]
        fig = dielectric_function.plot(direction)
        check_figure_contains_plots(fig, plots, Assert)


def test_dielectric_plot_real_or_imag(dielectric_function, Assert):
    real_plot = Plot(
        x=dielectric_function.ref.energies,
        y=isotropic(dielectric_function.ref.density_density).real,
        name=expected_plot_name("density", "Re", "isotropic"),
    )
    imag_plot = Plot(
        x=dielectric_function.ref.energies,
        y=isotropic(dielectric_function.ref.density_density).imag,
        name=expected_plot_name("density", "Im", "isotropic"),
    )
    for real in ("real", "Re"):
        fig = dielectric_function.plot(real)
        check_figure_contains_plots(fig, [real_plot], Assert)
    for imag in ("imaginary", "imag", "Im"):
        fig = dielectric_function.plot(imag)
        check_figure_contains_plots(fig, [imag_plot], Assert)


def test_dielectric_plot_nested(dielectric_function, Assert):
    plots = [
        Plot(
            x=dielectric_function.ref.energies,
            y=get_direction(dielectric_function.ref.density_density, "xx").real,
            name=expected_plot_name("density", "Re", "xx"),
        ),
        Plot(
            x=dielectric_function.ref.energies,
            y=get_direction(dielectric_function.ref.current_current, "xy").imag,
            name=expected_plot_name("current", "Im", "xy"),
        ),
        Plot(
            x=dielectric_function.ref.energies,
            y=get_direction(dielectric_function.ref.current_current, "yz").imag,
            name=expected_plot_name("current", "Im", "yz"),
        ),
        Plot(
            x=dielectric_function.ref.energies,
            y=get_direction(dielectric_function.ref.ion, "zz").real,
            name=expected_plot_name("ion", "Re", "zz"),
        ),
        Plot(
            x=dielectric_function.ref.energies,
            y=get_direction(dielectric_function.ref.ion, "zz").imag,
            name=expected_plot_name("ion", "Im", "zz"),
        ),
        Plot(
            x=dielectric_function.ref.energies,
            y=isotropic(dielectric_function.ref.density_density).real,
            name=expected_plot_name("density", "Re", "isotropic"),
        ),
        Plot(
            x=dielectric_function.ref.energies,
            y=isotropic(dielectric_function.ref.current_current).real,
            name=expected_plot_name("current", "Re", "isotropic"),
        ),
    ]
    selection = "density(Re(xx)) Im(current(xy,yz)) ion(zz(Re,Im)) Re(density,current)"
    fig = dielectric_function.plot(selection)
    check_figure_contains_plots(fig, plots, Assert)


def isotropic(tensor):
    return np.trace(tensor) / 3


def get_direction(tensor, direction):
    lookup = {"x": 0, "y": 1, "z": 2}
    i = lookup[direction[0]]
    j = lookup[direction[1]]
    return 0.5 * (tensor[i, j] + tensor[j, i])


def expected_plot_name(component, real_or_imag, direction):
    subscript = "" if direction == "isotropic" else f"_{{{direction}}}"
    if component == "density":
        superscript = "^{dd}"
    elif component == "current":
        superscript = "^{jj}"
    else:
        superscript = "^{ion}"
    return f"{real_or_imag}($\\epsilon{superscript}{subscript}$)"


def check_figure_contains_plots(fig, references, Assert):
    assert fig.layout.xaxis.title.text == "Energy (eV)"
    assert fig.layout.yaxis.title.text == r"$\epsilon$"
    assert len(fig.data) == len(references)
    for data, ref in zip(fig.data, references):
        Assert.allclose(data.x, ref.x)
        Assert.allclose(data.y, ref.y)
        assert data.name == ref.name


#
# def test_dielectric_to_image(dielectric):
#     check_to_image(dielectric, None, "dielectric.png")
#     custom_filename = "custom.jpg"
#     check_to_image(dielectric, custom_filename, custom_filename)
#
#
# def check_to_image(dielectric, filename_argument, expected_filename):
#     with patch("py4vasp.data.dielectric.Dielectric._to_plotly") as plot:
#         dielectric.to_image("args", filename=filename_argument, key="word")
#         plot.assert_called_once()
#         assert plot.call_args[0][1] == "args"
#         assert plot.call_args[1] == {"key": "word"}
#         fig = plot.return_value
#         fig.write_image.assert_called_once_with(dielectric._path / expected_filename)
#
#
# def test_dielectric_print(dielectric, format_):
#     actual, _ = format_(dielectric)
#     reference = f"""
# dielectric function:
#     energies: [0.00, 1.00] 50 points
#     directions: isotropic, xx, yy, zz, xy, yz, xz
#     """.strip()
#     assert actual == {"text/plain": reference}
#
#
def test_descriptor(dielectric_function, check_descriptors):
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_to_plotly": ["to_plotly", "plot"],
    }
    check_descriptors(dielectric_function, descriptors)


def test_from_file(raw_data, mock_file, check_read):
    raw_dielectric = raw_data.dielectric_function("default")
    with mock_file("dielectric_function", raw_dielectric) as mocks:
        check_read(DielectricFunction, mocks, raw_dielectric)
