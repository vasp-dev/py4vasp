# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import types
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation import QUANTITIES, Calculation
from py4vasp._calculation.optics import Optics, OpticsHandler

HBAR_C = 1239.84  # eV·nm


def _reflectivity(eps):
    n = np.sqrt(eps)
    return np.abs((n - 1) / (n + 1)) ** 2


def _absorption(eps, energies):
    k = np.sqrt(eps).imag
    alpha = 2 * k * energies / HBAR_C
    return alpha / np.max(alpha)


def _transmission(eps, energies):
    return np.clip(1 - _reflectivity(eps) - _absorption(eps, energies), 0, 1)


def isotropic(tensor):
    return np.trace(tensor) / 3


def get_direction(tensor, direction):
    lookup = {"x": 0, "y": 1, "z": 2}
    i = lookup[direction[0]]
    j = lookup[direction[1]]
    return 0.5 * (tensor[i, j] + tensor[j, i])


@pytest.fixture
def electron(raw_data):
    raw_dielectric = raw_data.dielectric_function("electron")
    optics = Optics.from_data(raw_dielectric)
    optics.ref = types.SimpleNamespace()
    optics.ref.raw_data = raw_dielectric
    optics.ref.energies = raw_dielectric.energies
    to_complex = lambda data: data[..., 0] + 1j * data[..., 1]
    optics.ref.dielectric_function = to_complex(raw_dielectric.dielectric_function)
    return optics


def test_optics_is_registered_and_resolves():
    assert "optics" in QUANTITIES
    calc = Calculation.from_path(".")
    assert isinstance(calc.optics, Optics)


def test_from_data_creates_dispatcher(electron):
    assert isinstance(electron, Optics)


def test_selections(electron):
    assert electron.selections() == {
        "directions": ["isotropic", "xx", "yy", "zz", "xy", "xz", "yz"]
    }


def check_coefficients(actual, eps, energies, Assert):
    Assert.allclose(actual["reflectivity"], _reflectivity(eps))
    Assert.allclose(actual["absorption"], _absorption(eps, energies))
    Assert.allclose(actual["transmission"], _transmission(eps, energies))


def test_read_default(electron, Assert):
    energies = electron.ref.energies
    eps = isotropic(electron.ref.dielectric_function)
    for method in (electron.read, electron.to_dict):
        actual = method()
        Assert.allclose(actual["energies"], energies)
        check_coefficients(actual, eps, energies, Assert)


def test_read_direction(electron, Assert):
    energies = electron.ref.energies
    for direction in ("xx", "yy", "zz", "xy", "xz", "yz"):
        eps = get_direction(electron.ref.dielectric_function, direction)
        check_coefficients(electron.read(direction), eps, energies, Assert)


def test_read_combined_selection(electron, Assert):
    # "xx + yy" combines the two directions into a single spectrum via the Selector
    energies = electron.ref.energies
    tensor = electron.ref.dielectric_function
    eps = get_direction(tensor, "xx") + get_direction(tensor, "yy")
    check_coefficients(electron.read("xx + yy"), eps, energies, Assert)


def test_read_list_selection(electron, Assert):
    # "xx, yy" yields two independent results keyed by direction label
    energies = electron.ref.energies
    tensor = electron.ref.dielectric_function
    actual = electron.read("xx, yy")
    Assert.allclose(actual["energies"], energies)
    for direction in ("xx", "yy"):
        eps = get_direction(tensor, direction)
        check_coefficients(actual[direction], eps, energies, Assert)


def test_read_scalar_dielectric_function_raises_error(raw_data):
    optics = Optics.from_data(raw_data.dielectric_function("q_point"))
    with pytest.raises(exception.IncorrectUsage):
        optics.read()


def test_print(electron, format_):
    actual, _ = format_(electron)
    reference = """\
optics:
    energies: [0.00, 1.00] 50 points
    directions: isotropic, xx, yy, zz, xy, xz, yz"""
    assert actual == {"text/plain": reference}


@dataclasses.dataclass
class Plot:
    x: np.ndarray
    y: np.ndarray
    label: str


def check_graph(fig, references, ylabel, Assert):
    assert fig.xlabel == "Energy (eV)"
    assert fig.ylabel == ylabel
    assert len(fig.series) == len(references)
    for series, ref in zip(fig.series, references):
        Assert.allclose(series.x, ref.x)
        Assert.allclose(series.y, ref.y)
        assert series.label == ref.label


def test_reflectivity_graph(electron, Assert):
    energies = electron.ref.energies
    tensor = electron.ref.dielectric_function
    default = [Plot(energies, _reflectivity(isotropic(tensor)), "reflectivity")]
    check_graph(electron.reflectivity(), default, "reflectivity", Assert)
    xx = [Plot(energies, _reflectivity(get_direction(tensor, "xx")), "reflectivity_xx")]
    check_graph(electron.reflectivity("xx"), xx, "reflectivity", Assert)


def test_absorption_graph(electron, Assert):
    energies = electron.ref.energies
    tensor = electron.ref.dielectric_function
    default = [Plot(energies, _absorption(isotropic(tensor), energies), "absorption")]
    check_graph(electron.absorption(), default, "absorption", Assert)
    eps_xx = get_direction(tensor, "xx")
    xx = [Plot(energies, _absorption(eps_xx, energies), "absorption_xx")]
    check_graph(electron.absorption("xx"), xx, "absorption", Assert)


def test_transmission_graph(electron, Assert):
    energies = electron.ref.energies
    tensor = electron.ref.dielectric_function
    default = [Plot(energies, _transmission(isotropic(tensor), energies), "transmission")]
    check_graph(electron.transmission(), default, "transmission", Assert)
    eps_xx = get_direction(tensor, "xx")
    xx = [Plot(energies, _transmission(eps_xx, energies), "transmission_xx")]
    check_graph(electron.transmission("xx"), xx, "transmission", Assert)


def _merged_plots(eps, energies, suffix=""):
    return [
        Plot(energies, _transmission(eps, energies), f"transmission{suffix}"),
        Plot(energies, _absorption(eps, energies), f"absorption{suffix}"),
        Plot(energies, _reflectivity(eps), f"reflectivity{suffix}"),
    ]


def test_plot_default(electron, Assert):
    energies = electron.ref.energies
    eps = isotropic(electron.ref.dielectric_function)
    refs = _merged_plots(eps, energies)
    check_graph(electron.plot(), refs, "coefficient", Assert)
    check_graph(electron.to_graph(), refs, "coefficient", Assert)


def test_plot_direction(electron, Assert):
    energies = electron.ref.energies
    eps = get_direction(electron.ref.dielectric_function, "xx")
    refs = _merged_plots(eps, energies, suffix="_xx")
    check_graph(electron.plot("xx"), refs, "coefficient", Assert)


def test_to_plotly(electron):
    with patch.object(Optics, "to_graph") as mock_graph:
        fig = electron.to_plotly("xx")
    mock_graph.assert_called_once_with("xx")
    assert fig == mock_graph.return_value.to_plotly.return_value


def check_to_image(optics, filename_argument, expected_filename):
    with patch.object(Optics, "to_plotly") as plot:
        optics.to_image("args", filename=filename_argument, key="word")
        plot.assert_called_once_with("args", key="word")
        fig = plot.return_value
        expected_path = optics.path / expected_filename
        fig.write_image.assert_called_once_with(expected_path)


def test_to_image(electron):
    check_to_image(electron, None, "optics.png")
    check_to_image(electron, "custom.jpg", "custom.jpg")
