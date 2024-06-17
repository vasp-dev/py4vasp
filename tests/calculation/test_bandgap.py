# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import calculation, exception

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
    return setup_bandgap(raw_gap)


@pytest.fixture
def spin_polarized(raw_data):
    raw_gap = raw_data.bandgap("spin_polarized")
    return setup_bandgap(raw_gap)


def setup_bandgap(raw_gap):
    bandgap = calculation.bandgap.from_data(raw_gap)
    bandgap.ref = types.SimpleNamespace()
    bandgap.ref.fundamental = raw_gap.values[..., CBM] - raw_gap.values[..., VBM]
    bandgap.ref.vbm = raw_gap.values[..., VBM]
    bandgap.ref.cbm = raw_gap.values[..., CBM]
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


def test_fundamental_default(bandgap, steps, Assert):
    actual = bandgap.fundamental() if steps == -1 else bandgap[steps].fundamental()
    Assert.allclose(actual, bandgap.ref.fundamental[steps, 0])


def test_fundamental_spin_polarized(spin_polarized, steps, Assert):
    bandgap = spin_polarized if steps == -1 else spin_polarized[steps]
    Assert.allclose(bandgap.fundamental(), bandgap.ref.fundamental[steps, 0])


def test_direct_default(bandgap, steps, Assert):
    actual = bandgap.direct() if steps == -1 else bandgap[steps].direct()
    Assert.allclose(actual, bandgap.ref.direct[steps, 0])


def test_fundamental_spin_polarized(spin_polarized, steps, Assert):
    bandgap = spin_polarized if steps == -1 else spin_polarized[steps]
    Assert.allclose(bandgap.direct(), bandgap.ref.direct[steps, 0])


def test_valence_band_maximum(bandgap, steps, Assert):
    if steps != -1:
        valence_band_maximum = bandgap[steps].valence_band_maximum()
    else:
        valence_band_maximum = bandgap.valence_band_maximum()
    Assert.allclose(valence_band_maximum, bandgap.ref.vbm[steps, 0])


def test_conduction_band_minimum(bandgap, steps, Assert):
    if steps != -1:
        conduction_band_minimum = bandgap[steps].conduction_band_minimum()
    else:
        conduction_band_minimum = bandgap.conduction_band_minimum()
    Assert.allclose(conduction_band_minimum, bandgap.ref.cbm[steps, 0])


def test_plot_default(bandgap, steps, Assert):
    graph = bandgap.plot() if steps == -1 else bandgap[steps].plot()
    check_default_graph(bandgap, steps, Assert, graph)


def test_plot_spin_polarized(spin_polarized, steps, Assert):
    graph = spin_polarized.plot() if steps == -1 else spin_polarized[steps].plot()
    check_default_graph(spin_polarized, steps, Assert, graph)


def check_default_graph(bandgap, steps, Assert, graph):
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


def test_plot_selection_default(bandgap, steps, Assert):
    graph = bandgap.plot("direct") if steps == -1 else bandgap[steps].plot("direct")
    assert len(graph.series) == 1
    assert graph.series[0].name == "direct"
    Assert.allclose(graph.series[0].y, bandgap.ref.direct[steps, 0])


def test_plot_selection_spin_polarized(spin_polarized, steps, Assert):
    bandgap = spin_polarized if steps == -1 else spin_polarized[steps]
    selection = "up, fundamental(down), independent(direct)"
    graph = bandgap.plot(selection)
    assert len(graph.series) == 4
    fundamental_up = graph.series[0]
    assert fundamental_up.name == "fundamental_up"
    Assert.allclose(fundamental_up.y, bandgap.ref.fundamental[steps, 1])
    direct_up = graph.series[1]
    assert direct_up.name == "direct_up"
    Assert.allclose(direct_up.y, bandgap.ref.direct[steps, 1])
    fundamental_down = graph.series[2]
    assert fundamental_down.name == "fundamental_down"
    Assert.allclose(fundamental_down.y, bandgap.ref.fundamental[steps, 2])
    direct = graph.series[3]
    assert direct.name == "direct"
    Assert.allclose(direct.y, bandgap.ref.direct[steps, 0])


@pytest.mark.parametrize("selection", ["up", "unknown", "fundamental(direct)"])
def test_plot_incorrect_selection(bandgap, selection):
    with pytest.raises(exception.IncorrectUsage):
        bandgap.plot(selection)


@patch("py4vasp._calculation.bandgap.Bandgap.to_graph")
def test_bandgap_to_plotly(mock_plot, bandgap):
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
    with patch("py4vasp._calculation.bandgap.Bandgap.to_plotly") as plot:
        bandgap.to_image("args", filename=filename_argument, key="word")
        plot.assert_called_once_with("args", key="word")
        fig = plot.return_value
        fig.write_image.assert_called_once_with(bandgap._path / expected_filename)


def test_print_default(bandgap, steps, format_):
    actual, _ = format_(bandgap) if steps == -1 else format_(bandgap[steps])
    reference = get_reference_string_default(steps)
    assert actual == {"text/plain": reference}


def get_reference_string_default(steps):
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


def test_print_spin_polarized(spin_polarized, steps, format_):
    bandgap = spin_polarized if steps == -1 else spin_polarized[steps]
    actual, _ = format_(bandgap)
    reference = get_reference_string_spin_polarized(steps)
    assert actual == {"text/plain": reference}


def get_reference_string_spin_polarized(steps):
    if steps == 0:
        return """\
Band structure
--------------
                       spin independent             spin component 1             spin component 2
val. band max:               0.000000                     3.741657                     5.291503
cond. band min:              1.000000                     3.872983                     5.385165
fundamental gap:             1.000000                     0.131326                     0.093662
VBM @ kpoint:       2.2361   2.4495   2.6458     4.3589   4.4721   4.5826     5.7446   5.8310   5.9161
CBM @ kpoint:       2.8284   3.0000   3.1623     4.6904   4.7958   4.8990     6.0000   6.0828   6.1644

lower band:                  1.414214                     4.000000                     5.477226
upper band:                  1.732051                     4.123106                     5.567764
direct gap:                  0.317837                     0.123106                     0.090539
@ kpoint:           3.3166   3.4641   3.6056     5.0000   5.0990   5.1962     6.2450   6.3246   6.4031

Fermi energy:                2.000000"""
    elif steps == slice(1, 3):
        return """\
Band structure
--------------
                       spin independent             spin component 1             spin component 2
val. band max:               9.165151                     9.899495                    10.583005
cond. band min:              9.219544                     9.949874                    10.630146
fundamental gap:             0.054393                     0.050379                     0.047141
VBM @ kpoint:       9.4340   9.4868   9.5394    10.1489  10.1980  10.2470    10.8167  10.8628  10.9087
CBM @ kpoint:       9.5917   9.6437   9.6954    10.2956  10.3441  10.3923    10.9545  11.0000  11.0454

lower band:                  9.273618                    10.000000                    10.677078
upper band:                  9.327379                    10.049876                    10.723805
direct gap:                  0.053761                     0.049876                     0.046727
@ kpoint:           9.7468   9.7980   9.8489    10.4403  10.4881  10.5357    11.0905  11.1355  11.1803

Fermi energy:                9.380832"""
    else:
        return """\
Band structure
--------------
                       spin independent             spin component 1             spin component 2
val. band max:              11.224972                    11.832160                    12.409674
cond. band min:             11.269428                    11.874342                    12.449900
fundamental gap:             0.044456                     0.042183                     0.040226
VBM @ kpoint:      11.4455  11.4891  11.5326    12.0416  12.0830  12.1244    12.6095  12.6491  12.6886
CBM @ kpoint:      11.5758  11.6190  11.6619    12.1655  12.2066  12.2474    12.7279  12.7671  12.8062

lower band:                 11.313708                    11.916375                    12.489996
upper band:                 11.357817                    11.958261                    12.529964
direct gap:                  0.044108                     0.041885                     0.039968
@ kpoint:          11.7047  11.7473  11.7898    12.2882  12.3288  12.3693    12.8452  12.8841  12.9228

Fermi energy:               11.401754"""


def test_factory_methods(raw_data, check_factory_methods):
    raw_gap = raw_data.bandgap("default")
    check_factory_methods(calculation.bandgap, raw_gap)
