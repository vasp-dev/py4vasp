from py4vasp.data import Kpoints
from py4vasp.exceptions import RefinementException
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


def test_mode(default_kpoints):
    test_kpoints = default_kpoints
    allowed_mode_formats = {
        "automatic": ["a", b"A", "auto"],
        "explicit": ["e", b"e", "explicit", "ExplIcIT"],
        "gamma": ["g", b"G", "gamma"],
        "line": ["l", b"l", "line"],
        "monkhorst": ["m", b"M", "  Monkhorst-Pack  "],
    }
    for mode, formats in allowed_mode_formats.items():
        for format in formats:
            test_kpoints.mode = format
            test_mode = Kpoints(test_kpoints).read()["mode"]
            assert test_mode == mode
    for unknown_mode in ["x", "y", "z", " "]:
        with pytest.raises(RefinementException):
            test_kpoints.mode = unknown_mode
            Kpoints(test_kpoints).read()
