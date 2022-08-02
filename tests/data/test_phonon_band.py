# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest
import types
from unittest.mock import patch
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
    Sr = slice(0, 2)
    band.ref.Sr = np.sum(np.abs(band.ref.modes[:, :, Sr, :]), axis=(2, 3))
    Ti = 2
    x = 0
    band.ref.Ti_x = np.abs(band.ref.modes[:, :, Ti, x])
    _45 = slice(3, 5)
    y = 1
    band.ref.y_45 = np.sum(np.abs(band.ref.modes[:, :, _45, y]), axis=2)
    z = 2
    band.ref.z = np.sum(np.abs(band.ref.modes[:, :, :, 2]), axis=2)
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
    checker = FatbandChecker(phonon_band, Assert)
    #
    default_width = 1
    graph = phonon_band.plot("Sr, 3(x), y(4:5), z")
    checker.verify(graph, default_width)
    #
    width = 0.25
    graph = phonon_band.plot("Sr, 3(x), y(4:5), z", width)
    checker.verify(graph, width)


class FatbandChecker:
    def __init__(self, phonon_band, Assert):
        ref = phonon_band.ref
        self.distances = ref.kpoints.distances()
        self.projections = ref.Sr, ref.Ti_x, ref.y_45, ref.z
        self.labels = "Sr", "Ti_1_x", "4:5_y", "z"
        self.bands = ref.bands
        self.Assert = Assert

    def verify(self, graph, width):
        for item in zip(graph.series, self.projections, self.labels):
            self.check_series(*item, width)

    def check_series(self, series, projection, label, width):
        assert series.name == label
        self.Assert.allclose(series.x, self.distances)
        self.Assert.allclose(series.y, self.bands.T)
        self.Assert.allclose(series.width, width * projection.T)


@patch("py4vasp.data._phonon_band.PhononBand.plot")
def test_to_plotly(mock_plot, phonon_band):
    fig = phonon_band.to_plotly("selection", width=0.2)
    mock_plot.assert_called_once_with("selection", 0.2)
    graph = mock_plot.return_value
    graph.to_plotly.assert_called_once()
    assert fig == graph.to_plotly.return_value


def test_to_image(phonon_band):
    check_to_image(phonon_band, None, "phonon_band.png")
    custom_filename = "custom.jpg"
    check_to_image(phonon_band, custom_filename, custom_filename)


def check_to_image(phonon_band, filename_argument, expected_filename):
    with patch("py4vasp.data._phonon_band.PhononBand.to_plotly") as plot:
        phonon_band.to_image("args", filename=filename_argument, key="word")
        plot.assert_called_once_with("args", key="word")
        fig = plot.return_value
        fig.write_image.assert_called_once_with(phonon_band._path / expected_filename)


def test_print(phonon_band, format_):
    actual, _ = format_(phonon_band)
    reference = """\
phonon band data:
    48 q-points
    21 modes
    Sr2TiO4"""
    assert actual == {"text/plain": reference}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.phonon_band("default")
    check_factory_methods(PhononBand, data)
