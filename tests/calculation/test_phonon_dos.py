# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp._calculation.phonon_dos import PhononDos


@pytest.fixture
def phonon_dos(raw_data):
    raw_dos = raw_data.phonon_dos("default")
    dos = PhononDos.from_data(raw_dos)
    dos.ref = types.SimpleNamespace()
    dos.ref.energies = raw_dos.energies
    dos.ref.total_dos = raw_dos.dos
    dos.ref.Sr = np.sum(raw_dos.projections[0:2], axis=(0, 1))
    dos.ref.Ti_x = raw_dos.projections[2, 0]
    dos.ref.y_45 = np.sum(raw_dos.projections[3:5, 1], axis=0)
    dos.ref.z = np.sum(raw_dos.projections[:, 2], axis=0)
    return dos


def test_phonon_dos_read(phonon_dos, Assert):
    actual = phonon_dos.read()
    Assert.allclose(actual["energies"], phonon_dos.ref.energies)
    Assert.allclose(actual["total"], phonon_dos.ref.total_dos)
    assert "Sr" not in actual


def test_phonon_dos_read_projection(phonon_dos, Assert):
    actual = phonon_dos.read("Sr, 3(x), y(4:5), z, Sr - Ti(x)")
    assert "total" in actual
    Assert.allclose(actual["Sr"], phonon_dos.ref.Sr)
    Assert.allclose(actual["Ti_1_x"], phonon_dos.ref.Ti_x)
    Assert.allclose(actual["4:5_y"], phonon_dos.ref.y_45)
    Assert.allclose(actual["z"], phonon_dos.ref.z)
    subtraction = phonon_dos.ref.Sr - phonon_dos.ref.Ti_x
    Assert.allclose(actual["Sr - Ti_x"], subtraction)


def test_phonon_dos_plot(phonon_dos, Assert):
    graph = phonon_dos.plot()
    assert graph.xlabel == "ω (THz)"
    assert graph.ylabel == "DOS (1/THz)"
    assert len(graph.series) == 1
    Assert.allclose(graph.series[0].x, phonon_dos.ref.energies)
    Assert.allclose(graph.series[0].y, phonon_dos.ref.total_dos)


def test_phonon_dos_plot_selection(phonon_dos, Assert):
    graph = phonon_dos.plot("Sr, 3(x), y(4:5), z")
    assert len(graph.series) == 5
    check_series(graph.series[0], phonon_dos.ref.total_dos, "total", Assert)
    check_series(graph.series[1], phonon_dos.ref.Sr, "Sr", Assert)
    check_series(graph.series[2], phonon_dos.ref.Ti_x, "Ti_1_x", Assert)
    check_series(graph.series[3], phonon_dos.ref.y_45, "4:5_y", Assert)
    check_series(graph.series[4], phonon_dos.ref.z, "z", Assert)


def check_series(series, reference, label, Assert):
    assert series.label == label
    Assert.allclose(series.y, reference)


@patch.object(PhononDos, "to_graph")
def test_phonon_dos_to_plotly(mock_plot, phonon_dos):
    fig = phonon_dos.to_plotly("selection")
    mock_plot.assert_called_once_with("selection")
    graph = mock_plot.return_value
    graph.to_plotly.assert_called_once()
    assert fig == graph.to_plotly.return_value


def test_phonon_dos_to_image(phonon_dos):
    check_to_image(phonon_dos, None, "phonon_dos.png")
    custom_filename = "custom.jpg"
    check_to_image(phonon_dos, custom_filename, custom_filename)


def check_to_image(phonon_dos, filename_argument, expected_filename):
    with patch.object(PhononDos, "to_plotly") as plot:
        phonon_dos.to_image("args", filename=filename_argument, key="word")
        plot.assert_called_once_with("args", key="word")
        fig = plot.return_value
        fig.write_image.assert_called_once_with(phonon_dos._path / expected_filename)


def test_selections(phonon_dos):
    assert phonon_dos.selections() == {
        "atom": ["Sr", "Ti", "O", "1", "2", "3", "4", "5", "6", "7"],
        "direction": ["x", "y", "z"],
    }


def test_phonon_dos_print(phonon_dos, format_):
    actual, _ = format_(phonon_dos)
    reference = """\
phonon DOS:
    [0.00, 5.00] mesh with 50 points
    21 modes
    Sr2TiO4"""
    assert actual == {"text/plain": reference}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.phonon_dos("default")
    check_factory_methods(PhononDos, data)
