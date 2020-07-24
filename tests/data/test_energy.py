from py4vasp.data import Energy
import pytest
import numpy as np
import py4vasp.raw as raw


@pytest.fixture
def reference_energy():
    labels = np.array(("ion-electron   TOTEN    ", "temperature    TEIN"), dtype="S")
    shape = (100, len(labels))
    return raw.Energy(labels=labels, values=np.arange(np.prod(shape)).reshape(shape))


def test_read_energy(reference_energy, Assert):
    conv = Energy(reference_energy)
    dict_ = conv.read()
    assert len(dict_) == 1
    label, data = dict_.popitem()
    assert label == reference_energy.labels[0].decode().strip()
    Assert.allclose(data, reference_energy.values[:, 0])
    label, data = conv.read("temperature").popitem()
    assert label == reference_energy.labels[1].decode().strip()
    Assert.allclose(data, reference_energy.values[:, 1])


def test_plot_energy(reference_energy, Assert):
    conv = Energy(reference_energy)
    fig = conv.plot()
    assert fig.layout.xaxis.title.text == "Step"
    assert fig.layout.yaxis.title.text == "Energy (eV)"
    Assert.allclose(fig.data[0].y, reference_energy.values[:, 0])
    fig = conv.plot("temperature")
    assert fig.layout.yaxis.title.text == "Temperature (K)"
    Assert.allclose(fig.data[0].y, reference_energy.values[:, 1])


def test_energy_from_file(reference_energy, mock_file, check_read):
    with mock_file("energy", reference_energy) as mocks:
        check_read(Energy, mocks, reference_energy)
