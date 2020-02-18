from py4vasp.data import Convergence
import pytest
import numpy as np
import py4vasp.raw as raw


@pytest.fixture
def reference_convergence():
    labels = np.array(("ion-electron   TOTEN    ", "temperature    TEIN"), dtype="S")
    shape = (100, len(labels))
    return raw.Convergence(
        labels=labels, energies=np.arange(np.prod(shape)).reshape(shape)
    )


def test_read_convergence(reference_convergence, Assert):
    conv = Convergence(reference_convergence)
    label, data = conv.read()
    assert label == reference_convergence.labels[0].decode().strip()
    Assert.allclose(data, reference_convergence.energies[:, 0])
    label, data = conv.read("temperature")
    assert label == reference_convergence.labels[1].decode().strip()
    Assert.allclose(data, reference_convergence.energies[:, 1])


def test_plot_convergence(reference_convergence, Assert):
    conv = Convergence(reference_convergence)
    fig = conv.plot()
    assert fig.layout.xaxis.title.text == "Step"
    assert fig.layout.yaxis.title.text == "Energy (eV)"
    Assert.allclose(fig.data[0].y, reference_convergence.energies[:, 0])
    fig = conv.plot("temperature")
    assert fig.layout.yaxis.title.text == "Temperature (K)"
    Assert.allclose(fig.data[0].y, reference_convergence.energies[:, 1])


def test_convergence_from_file(reference_convergence, mock_file, check_read):
    with mock_file("convergence", reference_convergence) as mocks:
        check_read(Convergence, mocks, reference_convergence)
