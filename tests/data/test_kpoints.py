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
        cell=raw.Cell(scale=1.0, lattice_vectors=np.eye(3)),
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
            test_mode = Kpoints(raw_kpoints).mode()
            assert test_mode == mode
    for unknown_mode in ["x", "y", "z", " "]:
        with pytest.raises(RefinementException):
            raw_kpoints.mode = unknown_mode
            Kpoints(raw_kpoints).mode()


def test_line_length(raw_kpoints):
    assert Kpoints(raw_kpoints).line_length() == len(raw_kpoints.coordinates)
    raw_kpoints.mode = "auto"
    raw_kpoints.number = 0  # automatic mode is indicated by setting number to 0
    assert Kpoints(raw_kpoints).line_length() == len(raw_kpoints.coordinates)
    set_line_mode(raw_kpoints)
    assert Kpoints(raw_kpoints).line_length() == raw_kpoints.number


def test_number_lines(raw_kpoints):
    assert Kpoints(raw_kpoints).number_lines() == 1
    set_line_mode(raw_kpoints)
    ref_number_lines = len(raw_kpoints.coordinates) / raw_kpoints.number
    assert Kpoints(raw_kpoints).number_lines() == ref_number_lines


def test_labels(raw_kpoints):
    raw_kpoints.labels = ["A", b"B", "C"]
    raw_kpoints.label_indices = [5, 8, 14]
    actual = Kpoints(raw_kpoints).labels()
    ref = [""] * len(raw_kpoints.coordinates)
    # note index difference between Fortran and Python
    ref[4] = "A"
    ref[7] = "B"
    ref[13] = "C"
    assert actual == ref
    set_line_mode(raw_kpoints)
    raw_kpoints.labels = ["W", "X", " Y ", "Z"]
    raw_kpoints.label_indices = [1, 2, 3, 6]
    actual = Kpoints(raw_kpoints).labels()
    ref = [""] * len(raw_kpoints.coordinates)
    ref[0] = "W"
    ref[raw_kpoints.number - 1] = "X"
    ref[raw_kpoints.number] = "Y"
    ref[3 * raw_kpoints.number - 1] = "Z"
    assert actual == ref


def test_distances_nontrivial_cell(raw_kpoints, Assert):
    cell = raw.Cell(
        scale=2.0, lattice_vectors=np.array([[3, 0, 0], [-1, 2, 0], [0, 0, 4]])
    )
    cartesian_kpoints = np.linspace(np.zeros(3), np.ones(3))
    direct_kpoints = cartesian_kpoints @ cell.lattice_vectors.T * cell.scale
    ref_dists = np.linalg.norm(cartesian_kpoints, axis=1)
    raw_kpoints.cell = cell
    raw_kpoints.coordinates = direct_kpoints
    dists = Kpoints(raw_kpoints).distances()
    Assert.allclose(dists, ref_dists)


def test_distances_lines(raw_kpoints, Assert):
    set_line_mode(raw_kpoints)
    first_line = np.linspace([0.5, 0.5, 0.5], [1, 0, 0], raw_kpoints.number)
    second_line = np.linspace([0, 1, 0], [0, 0, 0], raw_kpoints.number)
    raw_kpoints.coordinates = np.concatenate((first_line, second_line))
    first_dists = np.linalg.norm(first_line - first_line[0], axis=1)
    second_dists = np.linalg.norm(second_line - second_line[0], axis=1)
    second_dists += first_dists[-1]
    ref_dists = np.concatenate((first_dists, second_dists))
    dists = Kpoints(raw_kpoints).distances()
    Assert.allclose(dists, ref_dists)


def set_line_mode(kpoints):
    kpoints.mode = "line"
    kpoints.number = 5
