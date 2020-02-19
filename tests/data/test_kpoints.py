from py4vasp.data import Kpoints
import py4vasp.raw as raw
import pytest
import numpy as np


@pytest.fixture
def default_kpoints():
    number_kpoints = 20
    shape = (3, number_kpoints)
    return raw.Kpoints(
        mode="explicit",
        number=number_kpoints,
        coordinates=np.arange(np.prod(shape)).reshape(shape),
        weights=np.arange(number_kpoints),
    )


def test_read(default_kpoints, Assert):
    kpoints = Kpoints(default_kpoints)
    actual = kpoints.read()
    assert actual["mode"] == default_kpoints.mode
    Assert.allclose(actual["coordinates"], default_kpoints.coordinates)
    Assert.allclose(actual["weights"], default_kpoints.weights)
    assert actual["labels"] is None
