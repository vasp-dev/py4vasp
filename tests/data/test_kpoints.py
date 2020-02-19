from py4vasp.data import Kpoints
from py4vasp.exceptions import RefinementException
import py4vasp.raw as raw
import pytest
import numpy as np


@pytest.fixture
def raw_kpoints():
    number_kpoints = 20
    shape = (number_kpoints, 3)
    return raw.Kpoints(
        mode="explicit",
        number=number_kpoints,
        coordinates=np.arange(np.prod(shape)).reshape(shape),
        weights=np.arange(number_kpoints),
    )


def test_read(raw_kpoints, Assert):
    actual = Kpoints(raw_kpoints).read()
    assert actual["mode"] == raw_kpoints.mode
    Assert.allclose(actual["coordinates"], raw_kpoints.coordinates)
    Assert.allclose(actual["weights"], raw_kpoints.weights)
    assert actual["labels"] is None


def test_mode(raw_kpoints):
    allowed_mode_formats = {
        "automatic": ["a", b"A", "auto"],
        "explicit": ["e", b"e", "explicit", "ExplIcIT"],
        "gamma": ["g", b"G", "gamma"],
        "line": ["l", b"l", "line"],
        "monkhorst": ["m", b"M", "  Monkhorst-Pack  "],
    }
    for mode, formats in allowed_mode_formats.items():
        for format in formats:
            raw_kpoints.mode = format
            test_mode = Kpoints(raw_kpoints).read()["mode"]
            assert test_mode == mode
    for unknown_mode in ["x", "y", "z", " "]:
        with pytest.raises(RefinementException):
            raw_kpoints.mode = unknown_mode
            Kpoints(raw_kpoints).read()


def test_line_length(raw_kpoints):
    assert Kpoints(raw_kpoints).line_length() == len(raw_kpoints.coordinates)
    raw_kpoints.mode = "auto"
    raw_kpoints.number = 0  # automatic mode is indicated by setting number to 0
    assert Kpoints(raw_kpoints).line_length() == len(raw_kpoints.coordinates)
    raw_kpoints.mode = "line"
    raw_kpoints.number = 5
    assert Kpoints(raw_kpoints).line_length() == raw_kpoints.number


def test_number_lines(raw_kpoints):
    assert Kpoints(raw_kpoints).number_lines() == 1
    raw_kpoints.mode = "line"
    raw_kpoints.number = 5
    ref_number_lines = len(raw_kpoints.coordinates) / raw_kpoints.number
    assert Kpoints(raw_kpoints).number_lines() == ref_number_lines
