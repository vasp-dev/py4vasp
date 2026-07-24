# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import types
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception, raw
from py4vasp._calculation import QUANTITIES, Calculation
from py4vasp._calculation import _optics_color as color
from py4vasp._calculation.optics import Optics, OpticsHandler
from py4vasp._raw.definition import unique_selections
from py4vasp._util.color import Color

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


@pytest.fixture
def visible():
    # The demo energies span 0-1 eV (infrared); the color requires energies in the
    # visible range so we build a dedicated dielectric function here.
    rng = np.random.default_rng(1)
    energies = np.linspace(0.5, 4.0, 60)
    data = 10 * rng.standard_normal((3, 3, len(energies), 2))
    raw_dielectric = raw.DielectricFunction(
        energies=energies,
        dielectric_function=raw.VaspData(data),
        current_current=raw.VaspData(None),
    )
    optics = Optics.from_data(raw_dielectric)
    optics.ref = types.SimpleNamespace()
    optics.ref.raw_data = raw_dielectric
    optics.ref.energies = energies
    optics.ref.dielectric_function = data[..., 0] + 1j * data[..., 1]
    return optics


def test_optics_is_registered_and_resolves():
    assert "optics" in QUANTITIES
    calc = Calculation.from_path(".")
    assert isinstance(calc.optics, Optics)


def test_from_data_creates_dispatcher(electron):
    assert isinstance(electron, Optics)


def test_selections(electron):
    assert electron.selections() == {
        "optics": list(unique_selections("dielectric_function")),
        "components": ["transmission", "absorption", "reflectivity"],
        "directions": ["isotropic", "xx", "yy", "zz", "xy", "xz", "yz"],
    }
    # the sources include the schema entries the optics quantity derives from
    assert electron.selections()["optics"][0] == "default"
    assert "bse" in electron.selections()["optics"]


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
    eps = isotropic(tensor)
    default = [Plot(energies, _transmission(eps, energies), "transmission")]
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


def test_plot_single_component(electron, Assert):
    # selecting a coefficient plots only that coefficient and labels the axis with it
    energies = electron.ref.energies
    eps = isotropic(electron.ref.dielectric_function)
    expected = [Plot(energies, _transmission(eps, energies), "transmission")]
    check_graph(electron.plot("transmission"), expected, "transmission", Assert)


def test_plot_component_equivalent_to_method(electron):
    # plot("transmission") is equivalent to transmission()
    from_plot = electron.plot("transmission")
    from_method = electron.transmission()
    assert [s.label for s in from_plot.series] == [s.label for s in from_method.series]
    assert from_plot.ylabel == from_method.ylabel


def test_plot_multiple_components(electron, Assert):
    energies = electron.ref.energies
    eps = isotropic(electron.ref.dielectric_function)
    expected = [
        Plot(energies, _transmission(eps, energies), "transmission"),
        Plot(energies, _reflectivity(eps), "reflectivity"),
    ]
    fig = electron.plot("transmission, reflectivity")
    check_graph(fig, expected, "coefficient", Assert)


def test_plot_component_with_direction(electron, Assert):
    energies = electron.ref.energies
    eps = get_direction(electron.ref.dielectric_function, "xx")
    expected = [Plot(energies, _reflectivity(eps), "reflectivity_xx")]
    check_graph(electron.plot("reflectivity(xx)"), expected, "reflectivity", Assert)


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


def _reference_color(eps, energies, spectrum="reflectivity", **kwargs):
    if spectrum == "reflectivity":
        coefficient = _reflectivity(eps)
    else:
        coefficient = _transmission(eps, energies)
    mask = energies > 0
    wavelengths = HBAR_C / energies[mask]
    order = np.argsort(wavelengths)
    return color.spectrum_to_rgb(wavelengths[order], coefficient[mask][order], **kwargs)


def test_color_default(visible, Assert):
    pytest.importorskip("scipy")
    eps = isotropic(visible.ref.dielectric_function)
    expected = _reference_color(eps, visible.ref.energies)
    result = visible.color()
    assert isinstance(result, Color)
    assert result.label() == "reflectivity"
    Assert.allclose(result.rgb, expected)


def test_color_defaults_match_explicit(visible):
    pytest.importorskip("scipy")
    assert visible.color() == visible.color(illuminant="D65", cmf="1931_2")


def test_color_range(visible):
    pytest.importorskip("scipy")
    color = visible.color("xx")
    assert len(color.rgb) == 3
    assert all(0 <= channel <= 1 for channel in color.rgb)


def test_color_from_transmission(visible, Assert):
    pytest.importorskip("scipy")
    # the coefficient is now part of the selection, defaulting to reflectivity
    eps = isotropic(visible.ref.dielectric_function)
    expected = _reference_color(eps, visible.ref.energies, spectrum="transmission")
    result = visible.color("transmission")
    assert result.label() == "transmission"
    Assert.allclose(result.rgb, expected)


def test_color_component_switch_changes_result(visible):
    pytest.importorskip("scipy")
    reflectivity = visible.color("reflectivity").rgb
    transmission = visible.color("transmission").rgb
    assert not np.allclose(reflectivity, transmission)


def test_color_list_selection(visible):
    pytest.importorskip("scipy")
    result = visible.color("xx, yy")
    assert set(result) == {"reflectivity_xx", "reflectivity_yy"}
    assert all(isinstance(color, Color) for color in result.values())
    assert result["reflectivity_xx"].label() == "reflectivity_xx"


def test_color_multiple_components(visible, Assert):
    pytest.importorskip("scipy")
    result = visible.color("reflectivity, transmission")
    assert set(result) == {"reflectivity", "transmission"}
    Assert.allclose(result["transmission"].rgb, visible.color("transmission").rgb)


def test_color_invalid_illuminant_raises_error(visible):
    pytest.importorskip("scipy")
    # the illuminant is validated only after scipy interpolates the spectrum, so this
    # path requires the full (not core) installation
    with pytest.raises(exception.IncorrectUsage):
        visible.color(illuminant="does-not-exist")


def test_color_invalid_cmf_raises_error(visible):
    # the color matching function is validated before scipy is used, so this works on core
    with pytest.raises(exception.IncorrectUsage):
        visible.color(cmf="does-not-exist")


def test_to_database(visible, Assert):
    pytest.importorskip("scipy")
    from py4vasp._raw.models import OpticsModel

    handler = OpticsHandler.from_data(visible.ref.raw_data)
    db_data = handler.to_database()
    assert isinstance(db_data, OpticsModel)

    energies = visible.ref.energies
    eps = isotropic(visible.ref.dielectric_function)
    assert db_data.energy_min == float(np.min(energies))
    assert db_data.energy_max == float(np.max(energies))
    Assert.allclose(db_data.reflectivity_min, np.min(_reflectivity(eps)))
    Assert.allclose(db_data.reflectivity_max, np.max(_reflectivity(eps)))
    Assert.allclose(db_data.absorption_min, np.min(_absorption(eps, energies)))
    Assert.allclose(db_data.absorption_max, np.max(_absorption(eps, energies)))
    Assert.allclose(db_data.transmission_min, np.min(_transmission(eps, energies)))
    Assert.allclose(db_data.transmission_max, np.max(_transmission(eps, energies)))

    expected_color = Color(_reference_color(eps, energies))
    Assert.allclose(db_data.color_rgb, list(expected_color.rgb))
    assert db_data.color_hex == expected_color.hex
    # scalar fields are plain floats and the color is a fixed-size tuple / hex string
    assert all(
        isinstance(getattr(db_data, name), float)
        for name in ("energy_min", "reflectivity_max", "transmission_min")
    )
    assert isinstance(db_data.color_rgb, tuple)
    assert isinstance(db_data.color_hex, str)


def test_to_database_keyed_by_optics(visible):
    pytest.importorskip("scipy")
    from py4vasp._raw.models import OpticsModel

    result = visible._to_database()
    assert isinstance(result, dict)
    # the DataSource ignores the selection, so all sources collapse to a single "optics"
    assert "optics" in result
    assert isinstance(result["optics"], dict)
    assert isinstance(result["optics"]["default"], OpticsModel)
    # keys are derived from "optics", never from the underlying "dielectric_function"
    assert not any(key.startswith("dielectric_function") for key in result)


def test_to_database_scalar_dielectric_function_is_skipped(raw_data):
    optics = Optics.from_data(raw_data.dielectric_function("q_point"))
    # scalar dielectric functions cannot yield directional optics; database collection
    # must swallow the error and simply omit the quantity rather than propagate it
    assert optics._to_database() == {}


def test_factory_methods_read_dielectric_function(raw_data):
    # Optics owns no data of its own; from_path/from_file must access the dielectric
    # function in the schema rather than a nonexistent "optics" entry.
    data = raw_data.dielectric_function("electron")
    instances = (Optics.from_path(), Optics.from_file("vaspout.h5"))
    calls = (
        lambda optics: optics.read(),
        lambda optics: optics.reflectivity(),
        lambda optics: optics.selections(),
        lambda optics: str(optics),
    )
    for optics in instances:
        for call in calls:
            with patch("py4vasp.raw.access") as mock_access:
                mock_access.return_value.__enter__.return_value = data
                call(optics)
                mock_access.assert_called_once()
                assert mock_access.call_args.args[0] == "dielectric_function"


def test_is_available(raw_data):
    tensor = Optics.from_data(raw_data.dielectric_function("electron"))
    scalar = Optics.from_data(raw_data.dielectric_function("q_point"))
    assert tensor.is_available() is True
    # the optical spectra need the full tensor; a scalar dielectric function fails
    assert scalar.is_available() is False
    # selections works regardless of the tensor/scalar form
    assert scalar.is_available(method="selections") is True
