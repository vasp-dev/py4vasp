from py4vasp.data import Energy, _util
from py4vasp.raw import RawEnergy, RawVersion
from . import current_vasp_version
import pytest
import numpy as np
import py4vasp.exceptions as exception


number_time_step = 100


@pytest.fixture
def reference_energy():
    labels = ("ion-electron   TOTEN    ", "kinetic energy EKIN", "temperature    TEIN")
    labels = np.array(labels, dtype="S")
    shape = (number_time_step, len(labels))
    return RawEnergy(
        version=current_vasp_version,
        labels=labels,
        values=np.arange(np.prod(shape)).reshape(shape),
    )


def test_read_energy(reference_energy, Assert):
    default_label = reference_energy.labels[0].decode().strip()
    kinetic_label = reference_energy.labels[1].decode().strip()
    temperature_label = reference_energy.labels[2].decode().strip()
    energy = Energy(reference_energy)
    #
    dict_ = energy.read()
    assert len(dict_) == 1
    Assert.allclose(dict_[default_label], reference_energy.values[:, 0])
    #
    dict_ = energy.read("temperature")
    Assert.allclose(dict_[temperature_label], reference_energy.values[:, 2])
    #
    dict_ = energy.read("TOTEN, EKIN")
    assert len(dict_) == 2
    Assert.allclose(dict_[default_label], reference_energy.values[:, 0])
    Assert.allclose(dict_[kinetic_label], reference_energy.values[:, 1])


def test_plot_energy(reference_energy, Assert):
    energy = Energy(reference_energy)
    fig = energy.plot()
    assert fig.layout.xaxis.title.text == "Step"
    assert fig.layout.yaxis.title.text == "Energy (eV)"
    Assert.allclose(fig.data[0].x, np.arange(number_time_step) + 1)
    Assert.allclose(fig.data[0].y, reference_energy.values[:, 0])
    fig = energy.plot("temperature")
    assert fig.layout.yaxis.title.text == "Temperature (K)"
    assert "yaxis2" not in fig.layout
    Assert.allclose(fig.data[0].y, reference_energy.values[:, 2])
    fig = energy.plot("temperature, EKIN")
    assert fig.layout.yaxis.title.text == "Energy (eV)"
    assert fig.layout.yaxis2.title.text == "Temperature (K)"
    Assert.allclose(fig.data[0].y, reference_energy.values[:, 2])
    assert fig.data[0].name == "temperature"
    Assert.allclose(fig.data[1].y, reference_energy.values[:, 1])
    assert fig.data[1].name == "kinetic energy"


def test_final_energy(reference_energy, Assert):
    energy = Energy(reference_energy)
    Assert.allclose(energy.final(), reference_energy.values[-1, 0])
    Assert.allclose(energy.final("temperature"), reference_energy.values[-1, 2])
    E_total, E_kinetic = energy.final("TOTEN, EKIN")
    Assert.allclose(E_total, reference_energy.values[-1, 0])
    Assert.allclose(E_kinetic, reference_energy.values[-1, 1])


def test_energy_from_file(reference_energy, mock_file, check_read):
    with mock_file("energy", reference_energy) as mocks:
        check_read(Energy, mocks, reference_energy)


def test_print(reference_energy):
    actual, _ = _util.format_(Energy(reference_energy))
    reference = f"""
Energies at last step:
   ion-electron   TOTEN  =       297.000000
   kinetic energy EKIN   =       298.000000
   temperature    TEIN   =       299.000000
    """.strip()
    assert actual == {"text/plain": reference}


def test_incorrect_label(reference_energy):
    energy = Energy(reference_energy)
    with pytest.raises(exception.IncorrectUsage):
        energy.read("not available")
    with pytest.raises(exception.IncorrectUsage):
        energy.plot("not available")
    with pytest.raises(exception.IncorrectUsage):
        number_instead_of_string = 1
        energy.plot(number_instead_of_string)


def test_version(reference_energy):
    reference_energy.version = RawVersion(_util._minimal_vasp_version.major - 1)
    with pytest.raises(exception.OutdatedVaspVersion):
        Energy(reference_energy).read()


def test_descriptor(reference_energy, check_descriptors):
    energy = Energy(reference_energy)
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_to_plotly": ["to_plotly", "plot"],
        "_final": ["final"],
    }
    check_descriptors(energy, descriptors)
