from py4vasp.data import Energy
from unittest.mock import patch
import pytest
import numpy as np
import types
import py4vasp.exceptions as exception


number_time_step = 100


@pytest.fixture
def energy(raw_data):
    raw_energy = raw_data.energy("default")
    energy = Energy(raw_energy)
    energy.ref = types.SimpleNamespace()
    energy.ref.default_label = "ion-electron   TOTEN"
    energy.ref.total_energy = raw_energy.values[:, 0]
    energy.ref.kinetic_label = "kinetic energy EKIN"
    energy.ref.kinetic_energy = raw_energy.values[:, 1]
    energy.ref.temperature_label = "temperature    TEIN"
    energy.ref.temperature = raw_energy.values[:, 2]
    return energy


def test_read_default(energy, Assert):
    dict_ = energy.read()
    assert len(dict_) == 1
    Assert.allclose(dict_[energy.ref.default_label], energy.ref.total_energy)


def test_read_temperature(energy, Assert):
    dict_ = energy.read("temperature")
    Assert.allclose(dict_[energy.ref.temperature_label], energy.ref.temperature)


def test_read_two_energies(energy, Assert):
    dict_ = energy.read("TOTEN, EKIN")
    assert len(dict_) == 2
    Assert.allclose(dict_[energy.ref.default_label], energy.ref.total_energy)
    Assert.allclose(dict_[energy.ref.kinetic_label], energy.ref.kinetic_energy)


def test_plot_default(energy, Assert):
    fig = energy.plot()
    assert fig.layout.xaxis.title.text == "Step"
    assert fig.layout.yaxis.title.text == "Energy (eV)"
    Assert.allclose(fig.data[0].x, np.arange(len(energy.ref.total_energy)) + 1)
    Assert.allclose(fig.data[0].y, energy.ref.total_energy)


def test_plot_temperature(energy, Assert):
    fig = energy.plot("temperature")
    assert fig.layout.yaxis.title.text == "Temperature (K)"
    assert "yaxis2" not in fig.layout
    Assert.allclose(fig.data[0].y, energy.ref.temperature)


def test_plot_energy_and_temperature(energy, Assert):
    fig = energy.plot("temperature, EKIN")
    assert fig.layout.yaxis.title.text == "Energy (eV)"
    assert fig.layout.yaxis2.title.text == "Temperature (K)"
    Assert.allclose(fig.data[0].y, energy.ref.temperature)
    assert fig.data[0].name == "temperature"
    Assert.allclose(fig.data[1].y, energy.ref.kinetic_energy)
    assert fig.data[1].name == "kinetic energy"


def test_final_default(energy, Assert):
    Assert.allclose(energy.final(), energy.ref.total_energy[-1])


def test_final_temperature(energy, Assert):
    Assert.allclose(energy.final("temperature"), energy.ref.temperature[-1])


def test_final_two_energies(energy, Assert):
    E_total, E_kinetic = energy.final("TOTEN, EKIN")
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


def test_to_png(energy):
    filename = "image.png"
    with patch("py4vasp.data.energy._to_plotly") as plot:
        energy.to_png(filename, "args", key="word")
        plot.assert_called_once()
        # note: call_args.args[0] is the raw data
        assert plot.call_args.args[1] == "args"
        assert plot.call_args.kwargs == {"key": "word"}
        fig = plot.return_value
        fig.write_image.assert_called_once_with(filename)


def test_print(energy, format_):
    actual, _ = format_(energy)
    reference = f"""
Energies at last step:
   ion-electron   TOTEN  =         9.000000
   kinetic energy EKIN   =        10.000000
   temperature    TEIN   =        11.000000
    """.strip()
    assert actual == {"text/plain": reference}


def test_descriptor(energy, check_descriptors):
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_to_plotly": ["to_plotly", "plot"],
        "_to_png": ["to_png"],
        "_final": ["final"],
    }
    check_descriptors(energy, descriptors)


# def test_energy_from_file(reference_energy, mock_file, check_read):
#     with mock_file("energy", reference_energy) as mocks:
#         check_read(Energy, mocks, reference_energy)
