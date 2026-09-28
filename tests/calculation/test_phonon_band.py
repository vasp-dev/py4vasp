# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp._calculation._dispersion import DispersionHandler
from py4vasp._calculation._stoichiometry import Stoichiometry
from py4vasp._calculation.kpoint import Kpoint
from py4vasp._calculation.phonon_band import PhononBand, PhononBandHandler
from py4vasp._calculation.phonon_mode import PhononMode
from py4vasp._raw.models import PhononBandModel
from py4vasp._util import convert


@pytest.fixture
def phonon_band(raw_data):
    raw_band = raw_data.phonon_band("default")
    band = PhononBand.from_data(raw_band)
    band.ref = types.SimpleNamespace()
    # VASP reports the branches in THz; py4vasp converts them to an energy in eV
    band.ref.bands = raw_band.dispersion.eigenvalues / convert.EV_TO_THZ
    # the graph is drawn in meV, where a phonon spectrum reads naturally
    band.ref.plotted = band.ref.bands * convert.EV_TO_MEV
    band.ref.modes = convert.to_complex(raw_band.eigenvectors)
    raw_qpoints = raw_band.dispersion.kpoints
    band.ref.qpoints = Kpoint.from_data(raw_qpoints)
    raw_stoichiometry = raw_band.stoichiometry
    band.ref.stoichiometry = Stoichiometry.from_data(raw_stoichiometry)
    Sr = slice(0, 2)
    band.ref.Sr = np.sum(np.abs(band.ref.modes[:, :, Sr, :]), axis=(2, 3))
    Ti = 2
    x, y, z = range(3)
    band.ref.Ti_x = np.abs(band.ref.modes[:, :, Ti, x])
    _45 = slice(3, 5)
    band.ref.y_45 = np.sum(np.abs(band.ref.modes[:, :, _45, y]), axis=2)
    band.ref.z = np.sum(np.abs(band.ref.modes[:, :, :, z]), axis=2)
    band.ref.raw_data = raw_band
    return band


def test_read(phonon_band, Assert):
    band = phonon_band.read()
    Assert.allclose(band["qpoint_distances"], phonon_band.ref.qpoints.distances())
    assert band["qpoint_labels"] == phonon_band.ref.qpoints.labels()
    Assert.allclose(band["bands"], phonon_band.ref.bands)
    Assert.allclose(band["modes"], phonon_band.ref.modes)


def test_read_reports_an_unstable_mode_as_a_negative_energy(raw_data, Assert):
    # VASP marks an unstable mode with a negative frequency; phonon.band keeps that
    # convention so the branch can be drawn below zero, where phonon.mode instead
    # reports the same mode as an imaginary energy
    raw_band = raw_data.phonon_band("default")
    eigenvalues = np.array(raw_band.dispersion.eigenvalues)
    eigenvalues[0, 0] = -12.0
    raw_band.dispersion.eigenvalues = eigenvalues
    bands = PhononBand.from_data(raw_band).read()["bands"]
    assert not np.iscomplexobj(bands)
    Assert.allclose(bands[0, 0], -12.0 / convert.EV_TO_THZ)


def test_plot(phonon_band, Assert):
    graph = phonon_band.plot()
    assert graph.ylabel == "ω (meV)"
    assert len(graph.series) == 1
    assert graph.series[0].weight is None
    Assert.allclose(graph.series[0].x, phonon_band.ref.qpoints.distances())
    Assert.allclose(graph.series[0].y, phonon_band.ref.plotted.T)
    check_ticks(graph, phonon_band.ref.qpoints, Assert)
    assert tuple(graph.xticks.values()) == (r"$\Gamma$", "", r"M|$\Gamma$", "Y", "M")


def check_ticks(graph, qpoints, Assert):
    dists = qpoints.distances()
    xticks = (*dists[:: qpoints.line_length()], dists[-1])
    Assert.allclose(list(graph.xticks.keys()), np.array(xticks))


def test_plot_selection(phonon_band, Assert):
    checker = FatbandChecker(phonon_band, Assert)
    #
    default_width = 1
    graph = phonon_band.plot("Sr, 3(x), y(4:5), z, Sr - Ti(x)")
    checker.verify(graph, default_width)
    #
    width = 0.25
    graph = phonon_band.plot("Sr, 3(x), y(4:5), z, Sr - Ti(x)", width)
    checker.verify(graph, width)


class FatbandChecker:
    def __init__(self, phonon_band, Assert):
        ref = phonon_band.ref
        self.distances = ref.qpoints.distances()
        self.projections = ref.Sr, ref.Ti_x, ref.y_45, ref.z, ref.Sr - ref.Ti_x
        self.labels = "Sr", "Ti_1_x", "4:5_y", "z", "Sr - Ti_x"
        self.bands = ref.plotted
        self.Assert = Assert

    def verify(self, graph, weight):
        for item in zip(graph.series, self.projections, self.labels):
            self.check_series(*item, weight)

    def check_series(self, series, projection, label, weight):
        assert series.label == label
        self.Assert.allclose(series.x, self.distances)
        self.Assert.allclose(series.y, self.bands.T)
        self.Assert.allclose(series.weight, weight * projection.T)


@patch.object(PhononBand, "to_graph")
def test_to_plotly(mock_plot, phonon_band):
    fig = phonon_band.to_plotly("selection", width=0.2)
    mock_plot.assert_called_once_with("selection", width=0.2)
    graph = mock_plot.return_value
    graph.to_plotly.assert_called_once()
    assert fig == graph.to_plotly.return_value


def test_to_image(phonon_band):
    check_to_image(phonon_band, None, "phonon_band.png")
    custom_filename = "custom.jpg"
    check_to_image(phonon_band, custom_filename, custom_filename)


def check_to_image(phonon_band, filename_argument, expected_filename):
    with patch.object(PhononBand, "to_plotly") as plot:
        phonon_band.to_image("args", filename=filename_argument, key="word")
        plot.assert_called_once_with("args", key="word")
        fig = plot.return_value
        fig.write_image.assert_called_once_with(phonon_band._path / expected_filename)


def test_selections(phonon_band):
    assert phonon_band.selections() == {
        "atom": ["Sr", "Ti", "O", "1", "2", "3", "4", "5", "6", "7"],
        "direction": ["x", "y", "z"],
    }


def test_print(phonon_band, format_):
    actual, _ = format_(phonon_band)
    reference = """\
phonon band data:
    20 q-points
    21 modes
    Sr2TiO4"""
    assert actual == {"text/plain": reference}


def test_to_database(phonon_band):
    handler = PhononBandHandler.from_data(phonon_band.ref.raw_data)
    db_data = handler.to_database()
    assert isinstance(db_data, PhononBandModel)
    # dispersion (phonon frequencies) is folded into the phonon band model
    dispersion = DispersionHandler.from_data(
        phonon_band.ref.raw_data.dispersion
    ).to_database()
    assert db_data.eigenvalue_min == dispersion.eigenvalue_min / convert.EV_TO_THZ
    assert db_data.eigenvalue_max == dispersion.eigenvalue_max / convert.EV_TO_THZ


def test_band_and_mode_agree_on_the_same_dataset(raw_data, Assert):
    # phonon.band and phonon.mode("dispersion") read the very same HDF5 dataset. They
    # represent an unstable mode differently -- negative here, imaginary there -- but
    # the magnitude has to be one energy, which it was not while one of them was in THz.
    raw_mode = raw_data.phonon_mode("dispersion")
    raw_band = raw_data.phonon_band("default")
    raw_band.dispersion.eigenvalues = np.array(raw_mode.frequencies)
    bands = PhononBand.from_data(raw_band).read()["bands"]
    frequencies = PhononMode.from_data(raw_mode).frequencies()
    assert np.any(bands < 0)  # the fixture must exercise an unstable mode
    Assert.allclose(np.abs(bands), np.abs(frequencies))


def test_print_writes_to_stdout(phonon_band, capsys):
    assert phonon_band.print() is None
    assert capsys.readouterr().out == str(phonon_band) + "\n"


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.phonon_band("default")
    check_factory_methods(PhononBand, data)
