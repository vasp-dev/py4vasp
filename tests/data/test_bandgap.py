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
KPOINT_OPTICAL = slice(11, 14)


@pytest.fixture
def bandgap(raw_data):
    raw_gap = raw_data.bandgap("default")
    bandgap = data.Bandgap.from_data(raw_gap)
    bandgap.ref = types.SimpleNamespace()
    bandgap.ref.fundamental = raw_gap.values[:, 0, CBM] - raw_gap.values[:, 0, VBM]
    bandgap.ref.kpoint_vbm = raw_gap.values[:, 0, KPOINT_VBM]
    bandgap.ref.kpoint_cbm = raw_gap.values[:, 0, KPOINT_CBM]
    bandgap.ref.optical = raw_gap.values[:, 0, TOP] - raw_gap.values[:, 0, BOTTOM]
    bandgap.ref.kpoint_optical = raw_gap.values[:, 0, KPOINT_OPTICAL]
    bandgap.ref.fermi_energy = raw_gap.values[:, 0, FERMI]
    return bandgap


@pytest.fixture(params=[slice(None), slice(1, 3), 0, -1])
def steps(request):
    return request.param


def test_read(bandgap, steps, Assert):
    actual = bandgap.read() if steps == -1 else bandgap[steps].read()
    Assert.allclose(actual["fundamental"], bandgap.ref.fundamental[steps])
    Assert.allclose(actual["kpoint_VBM"], bandgap.ref.kpoint_vbm[steps])
    Assert.allclose(actual["kpoint_CBM"], bandgap.ref.kpoint_cbm[steps])
    Assert.allclose(actual["optical"], bandgap.ref.optical[steps])
    Assert.allclose(actual["kpoint_optical"], bandgap.ref.kpoint_optical[steps])
    Assert.allclose(actual["fermi_energy"], bandgap.ref.fermi_energy[steps])


def test_fundamental(bandgap, steps, Assert):
    actual = bandgap.fundamental() if steps == -1 else bandgap[steps].fundamental()
    Assert.allclose(actual, bandgap.ref.fundamental[steps])


def test_optical(bandgap, steps, Assert):
    actual = bandgap.optical() if steps == -1 else bandgap[steps].optical()
    Assert.allclose(actual, bandgap.ref.optical[steps])


def test_plot(bandgap, steps, Assert):
    graph = bandgap.plot() if steps == -1 else bandgap[steps].plot()
    xx = np.arange(len(bandgap.ref.fundamental))[steps] + 1
    assert graph.xlabel == "Step"
    assert graph.ylabel == "bandgap (eV)"
    assert len(graph.series) == 2
    fundamental = graph.series[0]
    assert fundamental.name == "fundamental"
    Assert.allclose(fundamental.x, xx)
    Assert.allclose(fundamental.y, bandgap.ref.fundamental[steps])
    optical = graph.series[1]
    assert optical.name == "optical"
    Assert.allclose(optical.x, xx)
    Assert.allclose(optical.y, bandgap.ref.optical[steps])


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
bandgap:
    step: 1
    fundamental:  1.000000
    optical:      0.317837
kpoint:
    val. band min:   2.236068   2.449490   2.645751
    cond. band max:  2.828427   3.000000   3.162278
    optical gap:     3.316625   3.464102   3.605551"""
    if steps == slice(1, 3):
        return """\
bandgap:
    step: 3
    fundamental:  0.093662
    optical:      0.090539
kpoint:
    val. band min:   5.744563   5.830952   5.916080
    cond. band max:  6.000000   6.082763   6.164414
    optical gap:     6.244998   6.324555   6.403124"""
    return """\
bandgap:
    step: 4
    fundamental:  0.076698
    optical:      0.074954
kpoint:
    val. band min:   6.855655   6.928203   7.000000
    cond. band max:  7.071068   7.141428   7.211103
    optical gap:     7.280110   7.348469   7.416198"""


def test_factory_methods(raw_data, check_factory_methods):
    raw_gap = raw_data.bandgap("default")
    check_factory_methods(data.Bandgap, raw_gap)
