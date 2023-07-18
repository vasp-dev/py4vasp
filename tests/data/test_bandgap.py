# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import data

VBM = 0
CBM = 1
BOTTOM = 2
TOP = 3
FERMI = 4
KPOINT_VBM = slice(5, 8)
KPOINT_CBM = slice(8, 11)
KPOINT_DIRECT = slice(11, 14)


@pytest.fixture
def bandgap(raw_data):
    raw_gap = raw_data.bandgap("default")
    for l, x in zip(raw_gap.labels, raw_gap.values.T):
        print(l, x[0])
    return setup_bandgap(raw_gap)


@pytest.fixture
def spin_polarized(raw_data):
    raw_gap = raw_data.bandgap("spin_polarized")
    return setup_bandgap(raw_gap)


def setup_bandgap(raw_gap):
    bandgap = data.Bandgap.from_data(raw_gap)
    bandgap.ref = types.SimpleNamespace()
    bandgap.ref.fundamental = raw_gap.values[..., CBM] - raw_gap.values[..., VBM]
    bandgap.ref.kpoint_vbm = raw_gap.values[..., KPOINT_VBM]
    bandgap.ref.kpoint_cbm = raw_gap.values[..., KPOINT_CBM]
    bandgap.ref.direct = raw_gap.values[..., TOP] - raw_gap.values[..., BOTTOM]
    bandgap.ref.kpoint_direct = raw_gap.values[..., KPOINT_DIRECT]
    bandgap.ref.fermi_energy = raw_gap.values[:, 0, FERMI]
    return bandgap


@pytest.fixture(params=[slice(None), slice(1, 3), 0, -1])
def steps(request):
    return request.param


def test_read_default(bandgap, steps, Assert):
    actual = bandgap.read() if steps == -1 else bandgap[steps].read()
    Assert.allclose(actual["fundamental"], bandgap.ref.fundamental[steps, 0])
    Assert.allclose(actual["kpoint_VBM"], bandgap.ref.kpoint_vbm[steps, 0])
    Assert.allclose(actual["kpoint_CBM"], bandgap.ref.kpoint_cbm[steps, 0])
    Assert.allclose(actual["direct"], bandgap.ref.direct[steps, 0])
    Assert.allclose(actual["kpoint_direct"], bandgap.ref.kpoint_direct[steps, 0])
    Assert.allclose(actual["fermi_energy"], bandgap.ref.fermi_energy[steps])


def test_read_spin_polarized(spin_polarized, steps, Assert):
    actual = spin_polarized.read() if steps == -1 else spin_polarized[steps].read()
    ref = spin_polarized.ref
    for i, suffix in enumerate(("", "_up", "_down")):
        Assert.allclose(actual[f"fundamental{suffix}"], ref.fundamental[steps, i])
        Assert.allclose(actual[f"kpoint_VBM{suffix}"], ref.kpoint_vbm[steps, i])
        Assert.allclose(actual[f"kpoint_CBM{suffix}"], ref.kpoint_cbm[steps, i])
        Assert.allclose(actual[f"direct{suffix}"], ref.direct[steps, i])
        Assert.allclose(actual[f"kpoint_direct{suffix}"], ref.kpoint_direct[steps, i])
    Assert.allclose(actual["fermi_energy"], ref.fermi_energy[steps])


def test_fundamental(bandgap, steps, Assert):
    actual = bandgap.fundamental() if steps == -1 else bandgap[steps].fundamental()
    Assert.allclose(actual, bandgap.ref.fundamental[steps, 0])


def test_direct(bandgap, steps, Assert):
    actual = bandgap.direct() if steps == -1 else bandgap[steps].direct()
    Assert.allclose(actual, bandgap.ref.direct[steps, 0])


def test_plot(bandgap, steps, Assert):
    graph = bandgap.plot() if steps == -1 else bandgap[steps].plot()
    xx = np.arange(len(bandgap.ref.fundamental))[steps] + 1
    assert graph.xlabel == "Step"
    assert graph.ylabel == "bandgap (eV)"
    assert len(graph.series) == 2
    fundamental = graph.series[0]
    assert fundamental.name == "fundamental"
    Assert.allclose(fundamental.x, xx)
    Assert.allclose(fundamental.y, bandgap.ref.fundamental[steps, 0])
    direct = graph.series[1]
    assert direct.name == "direct"
    Assert.allclose(direct.x, xx)
    Assert.allclose(direct.y, bandgap.ref.direct[steps, 0])


@patch("py4vasp._data.bandgap.Bandgap.to_graph")
def test_energy_to_plotly(mock_plot, bandgap):
    fig = bandgap.to_plotly()
    mock_plot.assert_called_once_with()
    graph = mock_plot.return_value
    graph.to_plotly.assert_called_once()
    assert fig == graph.to_plotly.return_value


def test_to_image(bandgap):
    check_to_image(bandgap, None, "bandgap.png")
    custom_filename = "custom.jpg"
    check_to_image(bandgap, custom_filename, custom_filename)


def check_to_image(bandgap, filename_argument, expected_filename):
    with patch("py4vasp._data.bandgap.Bandgap.to_plotly") as plot:
        bandgap.to_image("args", filename=filename_argument, key="word")
        plot.assert_called_once_with("args", key="word")
        fig = plot.return_value
        fig.write_image.assert_called_once_with(bandgap._path / expected_filename)


def test_print(bandgap, steps, format_):
    actual, _ = format_(bandgap) if steps == -1 else format_(bandgap[steps])
    reference = get_reference_string(steps)
    assert actual == {"text/plain": reference}


def get_reference_string(steps):
    if steps == 0:
        return """\
Band structure
--------------
                       spin independent
val. band max:               0.000000
cond. band min:              1.000000
fundamental gap:             1.000000
VBM @ kpoint:       2.2361   2.4495   2.6458
CBM @ kpoint:       2.8284   3.0000   3.1623

lower band:                  1.414214
upper band:                  1.732051
direct gap:                  0.317837
@ kpoint:           3.3166   3.4641   3.6056

Fermi energy:                2.000000"""
    elif steps == slice(1, 3):
        return """\
Band structure
--------------
                       spin independent
val. band max:               5.291503
cond. band min:              5.385165
fundamental gap:             0.093662
VBM @ kpoint:       5.7446   5.8310   5.9161
CBM @ kpoint:       6.0000   6.0828   6.1644

lower band:                  5.477226
upper band:                  5.567764
direct gap:                  0.090539
@ kpoint:           6.2450   6.3246   6.4031

Fermi energy:                5.656854"""
    else:
        return """\
Band structure
--------------
                       spin independent
val. band max:               6.480741
cond. band min:              6.557439
fundamental gap:             0.076698
VBM @ kpoint:       6.8557   6.9282   7.0000
CBM @ kpoint:       7.0711   7.1414   7.2111

lower band:                  6.633250
upper band:                  6.708204
direct gap:                  0.074954
@ kpoint:           7.2801   7.3485   7.4162

Fermi energy:                6.782330"""


def test_factory_methods(raw_data, check_factory_methods):
    raw_gap = raw_data.bandgap("default")
    check_factory_methods(data.Bandgap, raw_gap)
