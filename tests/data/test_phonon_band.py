# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest
import types
from py4vasp.data import PhononBand, Kpoint, Topology
from py4vasp._util import convert


@pytest.fixture
def phonon_band(raw_data):
    raw_band = raw_data.phonon_band("default")
    band = PhononBand.from_data(raw_band)
    band.ref = types.SimpleNamespace()
    band.ref.bands = raw_band.dispersion.eigenvalues
    band.ref.modes = convert.to_complex(raw_band.eigenvectors)
    raw_qpoints = raw_band.dispersion.kpoints
    band.ref.kpoints = Kpoint.from_data(raw_qpoints)
    band.ref.topology = Topology.from_data(raw_band.topology)
    index = slice(6)  # Sr: atom=0 and 1, 3 directions each
    band.ref.Sr = np.sum(np.abs(band.ref.modes[:, index, :]), axis=1)
    index = [6]  # Ti: atom=2, x: direction=0 -> (2*3 + 0)
    band.ref.Ti_x = np.sum(np.abs(band.ref.modes[:, index, :]), axis=1)
    index = [10, 13]  # 4:5: atom=3 and 4, y: direction=1 -> (3*3 + 1), (4*3 + 1)
    band.ref.y_45 = np.sum(np.abs(band.ref.modes[:, index, :]), axis=1)
    index = slice(2, None, 3)  # all atoms, z: direction=2
    band.ref.z = np.sum(np.abs(band.ref.modes[:, index, :]), axis=1)
    return band


def test_read(phonon_band, Assert):
    band = phonon_band.read()
    Assert.allclose(band["bands"], phonon_band.ref.bands)
    Assert.allclose(band["modes"], phonon_band.ref.modes)


def test_plot(phonon_band, Assert):
    graph = phonon_band.plot()
    assert graph.ylabel == "ω (THz)"
    assert len(graph.series) == 1
    assert graph.series[0].width is None
    Assert.allclose(graph.series[0].x, phonon_band.ref.kpoints.distances())
    Assert.allclose(graph.series[0].y, phonon_band.ref.bands.T)


def test_plot_selection(phonon_band, Assert):
    default_width = 1
    graph = phonon_band.plot("Sr, 3(x), y(4:5), z")
    ref = phonon_band.ref
    assert len(graph.series) == 4
    assert graph.series[0].name == "Sr"
    check_data(graph.series[0], default_width, ref.bands, ref.Sr, Assert)
    assert graph.series[1].name == "Ti_1_x"
    check_data(graph.series[1], default_width, ref.bands, ref.Ti_x, Assert)
    assert graph.series[2].name == "4:5_y"
    check_data(graph.series[2], default_width, ref.bands, ref.y_45, Assert)
    assert graph.series[3].name == "z"
    check_data(graph.series[3], default_width, ref.bands, ref.z, Assert)
    assert False


def test_plot_selection_width(phonon_band, Assert):
    width = 0.25
    graph = phonon_band.plot("Sr, 3(x), y(4:5), z", width)
    ref = phonon_band.ref
    assert len(graph.series) == 4
    assert graph.series[0].name == "Sr"
    check_data(graph.series[0], width, ref.bands, ref.Sr, Assert)
    assert graph.series[1].name == "Ti_1_x"
    check_data(graph.series[1], width, ref.bands, ref.Ti_x, Assert)
    assert graph.series[2].name == "4:5_y"
    check_data(graph.series[2], width, ref.bands, ref.y_45, Assert)
    assert graph.series[3].name == "z"
    check_data(graph.series[3], width, ref.bands, ref.z, Assert)


def check_data(series, width, band, projection, Assert):
    assert len(series.x) == series.y.shape[-1] == series.width.shape[-1]
    Assert.allclose(series.y, band.T)
    Assert.allclose(series.width, width * projection.T)


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.phonon_band("default")
    check_factory_methods(PhononBand, data)
