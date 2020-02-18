from unittest.mock import patch
from py4vasp.data import Convergence
import py4vasp.raw as raw
import pytest
import numpy as np


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


def test_convergence_from_file(reference_convergence):
    reference = Convergence(reference_convergence)
    with patch.object(raw.File, "__init__", autospec=True, return_value=None) as init:
        with patch.object(
            raw.File, "convergence", autospec=True, return_value=reference_convergence
        ) as conv:
            with patch.object(raw.File, "close", autospec=True) as close:
                mocks = {"init": init, "conv": conv, "close": close}
                check_read_from_open_file(mocks, reference)
                check_read_from_default_file(mocks, reference)
                check_read_from_filename("test", mocks, reference)


def check_read_from_open_file(mocks, reference):
    with raw.File() as file:
        reset_mocks(mocks)
        with Convergence.from_file(file) as actual:
            assert actual._raw == reference._raw
        mocks["init"].assert_not_called()
        mocks["conv"].assert_called_once()
        mocks["close"].assert_not_called()


def check_read_from_default_file(mocks, reference):
    reset_mocks(mocks)
    with Convergence.from_file() as actual:
        assert actual._raw == reference._raw
    mocks["init"].assert_called_once()
    mocks["conv"].assert_called_once()
    mocks["close"].assert_called_once()


def check_read_from_filename(filename, mocks, reference):
    reset_mocks(mocks)
    with Convergence.from_file(filename) as actual:
        assert actual._raw == reference._raw
    mocks["init"].assert_called_once()
    mocks["conv"].assert_called_once()
    mocks["close"].assert_called_once()


def reset_mocks(mocks):
    for mock in mocks.values():
        mock.reset_mock()
