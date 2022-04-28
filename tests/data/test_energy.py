# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp.data import Energy
from unittest.mock import patch
import pytest
import numbers
import numpy as np
import types
import py4vasp.exceptions as exception
import py4vasp._util.selection as selection


@pytest.fixture
def energy(raw_data):
    raw_energy = raw_data.energy("default")
    energy = Energy(raw_energy)
    energy.ref = types.SimpleNamespace()
    energy.ref.total_label = "ion-electron   TOTEN"
    energy.ref.total_energy = raw_energy.values[:, 0]
    energy.ref.kinetic_label = "kinetic energy EKIN"
    energy.ref.kinetic_energy = raw_energy.values[:, 1]
    energy.ref.temperature_label = "temperature    TEIN"
    energy.ref.temperature = raw_energy.values[:, 2]
    return energy


def test_read_default(energy, Assert):
    for steps in (slice(None), slice(1, 3), 0):
        check_read_default(energy, energy[steps].read(), steps, Assert)
    check_read_default(energy, energy.read(), -1, Assert)


def check_read_default(energy, dict_, steps, Assert):
    assert len(dict_) == 3
    Assert.allclose(dict_[energy.ref.total_label], energy.ref.total_energy[steps])
    Assert.allclose(dict_[energy.ref.kinetic_label], energy.ref.kinetic_energy[steps])
    Assert.allclose(dict_[energy.ref.temperature_label], energy.ref.temperature[steps])


def test_read_temperature(energy, Assert):
    for steps in (slice(None), slice(1, 3), 0):
        check_read_temperature(energy, energy[steps].read("temperature"), steps, Assert)
    check_read_temperature(energy, energy.read("temperature"), -1, Assert)


def check_read_temperature(energy, dict_, steps, Assert):
    assert len(dict_) == 1
    Assert.allclose(dict_[energy.ref.temperature_label], energy.ref.temperature[steps])


def test_read_two(energy, Assert):
    for steps in (slice(None), slice(1, 3), 0):
        check_read_two(energy, energy[steps].read("TOTEN, EKIN"), steps, Assert)
    check_read_two(energy, energy.read("TOTEN, EKIN"), -1, Assert)


def check_read_two(energy, dict_, steps, Assert):
    assert len(dict_) == 2
    Assert.allclose(dict_[energy.ref.total_label], energy.ref.total_energy[steps])
    Assert.allclose(dict_[energy.ref.kinetic_label], energy.ref.kinetic_energy[steps])


def test_plot_default(energy, Assert):
    for steps in (slice(None), slice(1, 3)):
        check_plot_default(energy, steps, Assert)


def check_plot_default(energy, steps, Assert):
    fig = energy[steps].plot()
    assert fig.xlabel == "Step"
    assert fig.ylabel == "Energy (eV)"
    Assert.allclose(fig.series[0].x, np.arange(len(energy.ref.total_energy))[steps] + 1)
    Assert.allclose(fig.series[0].y, energy.ref.total_energy[steps])


def test_plot_temperature(energy, Assert):
    for steps in (slice(None), slice(1, 3), 0):
        check_plot_temperature(energy, steps, Assert)


def check_plot_temperature(energy, steps, Assert):
    fig = energy[steps].plot("temperature")
    assert fig.ylabel == "Temperature (K)"
    assert fig.y2label is None
    Assert.allclose(fig.series[0].y, energy.ref.temperature[steps])


def test_plot_energy_and_temperature(energy, Assert):
    for steps in (slice(None), slice(1, 3)):
        check_plot_energy_and_temperature(energy, steps, Assert)


def check_plot_energy_and_temperature(energy, steps, Assert):
    fig = energy[steps].plot("temperature, EKIN")
    assert fig.ylabel == "Energy (eV)"
    assert fig.y2label == "Temperature (K)"
    Assert.allclose(fig.series[0].y, energy.ref.temperature[steps])
    assert fig.series[0].name == "temperature"
    Assert.allclose(fig.series[1].y, energy.ref.kinetic_energy[steps])
    assert fig.series[1].name == "kinetic energy"


def test_plot_all(energy, Assert):
    for steps in (slice(None), slice(1, 3)):
        check_plot_all(energy, steps, Assert)


def check_plot_all(energy, steps, Assert):
    fig = energy[steps].plot(selection.all)
    assert fig.ylabel == "Energy (eV)"
    assert fig.y2label == "Temperature (K)"
    Assert.allclose(fig.series[0].y, energy.ref.total_energy[steps])
    assert fig.series[0].name == "ion-electron"
    Assert.allclose(fig.series[1].y, energy.ref.kinetic_energy[steps])
    assert fig.series[1].name == "kinetic energy"
    Assert.allclose(fig.series[2].y, energy.ref.temperature[steps])
    assert fig.series[2].name == "temperature"


def test_to_numpy_default(energy, Assert):
    for steps in (slice(None), slice(1, 3), 0):
        actual = energy[steps].to_numpy()
        if isinstance(steps, slice):
            assert isinstance(actual, np.ndarray)
        Assert.allclose(actual, energy.ref.total_energy[steps])
    actual = energy.to_numpy()
    Assert.allclose(actual, energy.ref.total_energy[-1])
    assert isinstance(actual, numbers.Real)


def test_to_numpy_temperature(energy, Assert):
    for steps in (slice(None), slice(1, 3), 0):
        actual = energy[steps].to_numpy("temperature")
        Assert.allclose(actual, energy.ref.temperature[steps])
    Assert.allclose(energy.to_numpy("temperature"), energy.ref.temperature[-1])


def test_to_numpy_two_energies(energy, Assert):
    for steps in (slice(None), slice(1, 3), 0):
        E_total, E_kinetic = energy[steps].to_numpy("TOTEN, EKIN")
        Assert.allclose(E_total, energy.ref.total_energy[steps])
        Assert.allclose(E_kinetic, energy.ref.kinetic_energy[steps])
    E_total, E_kinetic = energy.to_numpy("TOTEN, EKIN")
    Assert.allclose(E_total, energy.ref.total_energy[-1])
    Assert.allclose(E_kinetic, energy.ref.kinetic_energy[-1])


def test_incorrect_label(energy):
    with pytest.raises(exception.IncorrectUsage):
        energy.read("not available")
    with pytest.raises(exception.IncorrectUsage):
        energy.plot("not available")
    with pytest.raises(exception.IncorrectUsage):
        number_instead_of_string = 1
        energy.plot(number_instead_of_string)


@patch("py4vasp.data.energy.Energy._plot")
def test_energy_to_plotly(mock_plot, energy):
    fig = energy.to_plotly("selection")
    mock_plot.assert_called_once_with("selection")
    graph = mock_plot.return_value
    graph.to_plotly.assert_called_once()
    assert fig == graph.to_plotly.return_value


def test_to_image(energy):
    check_to_image(energy, None, "energy.png")
    custom_filename = "custom.jpg"
    check_to_image(energy, custom_filename, custom_filename)


def check_to_image(energy, filename_argument, expected_filename):
    with patch("py4vasp.data.energy.Energy._to_plotly") as plot:
        energy.to_image("args", filename=filename_argument, key="word")
        plot.assert_called_once()
        # note: call_args[0][0] is self
        assert plot.call_args[0][1] == "args"
        assert plot.call_args[1] == {"key": "word"}
        fig = plot.return_value
        fig.write_image.assert_called_once_with(energy._path / expected_filename)


def test_labels(energy):
    total_energy = energy.ref.total_label
    kinetic_energy = energy.ref.kinetic_label
    temperature = energy.ref.temperature_label
    assert energy.labels() == [total_energy, kinetic_energy, temperature]
    assert energy.labels("temperature") == [temperature]
    assert energy.labels("TOTEN, EKIN") == [total_energy, kinetic_energy]


def test_print(energy, format_):
    actual, _ = format_(energy)
    check_print(actual, "final step", " 9.000000", "10.000000", "11.000000")
    actual, _ = format_(energy[0])
    check_print(actual, "step 1", " 0.000000", " 1.000000", " 2.000000")
    actual, _ = format_(energy[:])
    check_print(actual, "step 4 of range 1-4", " 9.000000", "10.000000", "11.000000")
    actual, _ = format_(energy[1:3])
    check_print(actual, "step 3 of range 2-3", " 6.000000", " 7.000000", " 8.000000")


def check_print(actual, step, toten, ekin, tein):
    reference = f"""
Energies at {step}:
   ion-electron   TOTEN  =        {toten}
   kinetic energy EKIN   =        {ekin}
   temperature    TEIN   =        {tein}
    """.strip()
    assert actual == {"text/plain": reference}


def test_descriptor(energy, check_descriptors):
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_plot": ["plot"],
        "_to_plotly": ["to_plotly"],
        "_to_numpy": ["to_numpy"],
    }
    check_descriptors(energy[-1], descriptors)


def test_from_file(raw_data, mock_file, check_read):
    raw_energy = raw_data.energy("default")
    with mock_file("energy", raw_energy) as mocks:
        check_read(Energy, mocks, raw_energy)
