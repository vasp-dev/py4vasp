# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numbers
import types
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._util import convert, select
from py4vasp.data import Energy


@pytest.fixture
def MD_energy(raw_data):
    raw_energy = raw_data.energy("MD")
    MD_energy = Energy.from_data(raw_energy)
    MD_energy.ref = types.SimpleNamespace()
    get_label = lambda x: convert.text_to_string(x).strip()
    MD_energy.ref.labels = [get_label(label) for label in raw_energy.labels]
    MD_energy.ref.values = raw_energy.values.T
    MD_energy.ref.total_label = "ion-electron   TOTEN"
    MD_energy.ref.total_energy = raw_energy.values[:, 0]
    MD_energy.ref.kinetic_label = "kinetic MD_energy EKIN"
    MD_energy.ref.kinetic_energy = raw_energy.values[:, 1]
    MD_energy.ref.temperature_label = "temperature    TEIN"
    MD_energy.ref.temperature = raw_energy.values[:, 3]
    return MD_energy


@pytest.mark.parametrize(
    "selection, subset",
    [
        (None, slice(None)),
        ("temperature", slice(3, 4)),
        ("TOTEN, EKIN", slice(None, 2)),
    ],
)
@pytest.mark.parametrize("steps", [slice(None), slice(1, 3), 0, -1])
def test_read(selection, subset, steps, MD_energy, Assert):
    kwargs = {"selection": selection} if selection else {}
    dict_ = MD_energy[steps].read(**kwargs) if steps != -1 else MD_energy.read(**kwargs)
    reference = MD_energy.ref
    assert len(dict_) == len(reference.labels[subset])
    for label, expected in zip(reference.labels[subset], reference.values[subset]):
        Assert.allclose(dict_[label], expected[steps])


def test_plot_default(MD_energy, Assert):
    for steps in (slice(None), slice(1, 3)):
        check_plot_default(MD_energy, steps, Assert)


def check_plot_default(MD_energy, steps, Assert):
    fig = MD_energy[steps].plot()
    assert fig.xlabel == "Step"
    assert fig.ylabel == "Energy (eV)"
    Assert.allclose(fig.series[0].x, np.arange(len(MD_energy.ref.values[0]))[steps] + 1)
    Assert.allclose(fig.series[0].y, MD_energy.ref.values[0, steps])


def test_plot_temperature(MD_energy, Assert):
    for steps in (slice(None), slice(1, 3), 0):
        check_plot_temperature(MD_energy, steps, Assert)


def check_plot_temperature(MD_energy, steps, Assert):
    fig = MD_energy[steps].plot("temperature")
    assert fig.ylabel == "Temperature (K)"
    assert fig.y2label is None
    Assert.allclose(fig.series[0].y, MD_energy.ref.values[3, steps])


def test_plot_energy_and_temperature(MD_energy, Assert):
    for steps in (slice(None), slice(1, 3)):
        check_plot_energy_and_temperature(MD_energy, steps, Assert)


def check_plot_energy_and_temperature(MD_energy, steps, Assert):
    fig = MD_energy[steps].plot("temperature, EKIN")
    assert fig.ylabel == "Energy (eV)"
    assert fig.y2label == "Temperature (K)"
    Assert.allclose(fig.series[0].y, MD_energy.ref.values[3, steps])
    assert fig.series[0].name == "temperature"
    Assert.allclose(fig.series[1].y, MD_energy.ref.values[1, steps])
    assert fig.series[1].name == "kinetic energy"


def test_to_numpy_default(MD_energy, Assert):
    for steps in (slice(None), slice(1, 3), 0):
        actual = MD_energy[steps].to_numpy()
        if isinstance(steps, slice):
            assert isinstance(actual, np.ndarray)
        Assert.allclose(actual, MD_energy.ref.values[0, steps])
    actual = MD_energy.to_numpy()
    Assert.allclose(actual, MD_energy.ref.values[0, -1])
    assert isinstance(actual, numbers.Real)


def test_to_numpy_temperature(MD_energy, Assert):
    for steps in (slice(None), slice(1, 3), 0):
        actual = MD_energy[steps].to_numpy("temperature")
        Assert.allclose(actual, MD_energy.ref.values[3, steps])
    Assert.allclose(MD_energy.to_numpy("temperature"), MD_energy.ref.values[3, -1])


def test_to_numpy_two_energies(MD_energy, Assert):
    for steps in (slice(None), slice(1, 3), 0):
        E_kin_lat, E_nose_pot = MD_energy[steps].to_numpy("EKIN_LAT, ES")
        Assert.allclose(E_kin_lat, MD_energy.ref.values[2, steps])
        Assert.allclose(E_nose_pot, MD_energy.ref.values[4, steps])
    E_kin_lat, E_nose_pot = MD_energy.to_numpy("EKIN_LAT, ES")
    Assert.allclose(E_kin_lat, MD_energy.ref.values[2, -1])
    Assert.allclose(E_nose_pot, MD_energy.ref.values[4, -1])


def test_incorrect_label(MD_energy):
    with pytest.raises(exception.IncorrectUsage):
        MD_energy.read("not available")
    with pytest.raises(exception.IncorrectUsage):
        MD_energy.plot("not available")
    with pytest.raises(exception.IncorrectUsage):
        number_instead_of_string = 1
        MD_energy.plot(number_instead_of_string)


@patch("py4vasp._data.energy.Energy.to_graph")
def test_energy_to_plotly(mock_plot, MD_energy):
    fig = MD_energy.to_plotly("selection")
    mock_plot.assert_called_once_with("selection")
    graph = mock_plot.return_value
    graph.to_plotly.assert_called_once()
    assert fig == graph.to_plotly.return_value


def test_to_image(MD_energy):
    check_to_image(MD_energy, None, "energy.png")
    custom_filename = "custom.jpg"
    check_to_image(MD_energy, custom_filename, custom_filename)


def check_to_image(MD_energy, filename_argument, expected_filename):
    with patch("py4vasp._data.energy.Energy.to_plotly") as plot:
        MD_energy.to_image("args", filename=filename_argument, key="word")
        plot.assert_called_once_with("args", key="word")
        fig = plot.return_value
        fig.write_image.assert_called_once_with(MD_energy._path / expected_filename)


def test_labels(MD_energy):
    total_energy = MD_energy.ref.labels[0]
    kinetic_energy = MD_energy.ref.labels[1]
    temperature = MD_energy.ref.labels[3]
    assert MD_energy.labels() == MD_energy.ref.labels
    assert MD_energy.labels("temperature") == [temperature]
    assert MD_energy.labels("TOTEN, EKIN") == [total_energy, kinetic_energy]


def test_print(MD_energy, format_):
    actual, _ = format_(MD_energy)
    check_print(actual, MD_energy.ref.labels, "final step", range(21, 28))
    actual, _ = format_(MD_energy[0])
    check_print(actual, MD_energy.ref.labels, "step 1", range(0, 7))
    actual, _ = format_(MD_energy[:])
    check_print(actual, MD_energy.ref.labels, "step 4 of range 1:4", range(21, 28))
    actual, _ = format_(MD_energy[1:3])
    check_print(actual, MD_energy.ref.labels, "step 3 of range 2:3", range(14, 20))


def check_print(actual, labels, step, energies):
    lines = [f"Energies at {step}:"]
    lines += [f"   {ll:23.23}={ee:17.6f}" for ll, ee in zip(labels, energies)]
    assert actual == {"text/plain": "\n".join(lines)}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.energy("MD")
    check_factory_methods(Energy, data)
