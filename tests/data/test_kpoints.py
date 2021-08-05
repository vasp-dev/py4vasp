from py4vasp.data import Kpoints
from py4vasp.raw import RawKpoints, RawVersion, RawCell
import py4vasp.exceptions as exception
import pytest
import numpy as np


@pytest.fixture
def raw_kpoints():
    number_kpoints = 20
    shape = (number_kpoints, 3)
    return RawKpoints(
        mode="explicit",
        number=number_kpoints,
        coordinates=np.arange(np.prod(shape)).reshape(shape),
        weights=np.arange(number_kpoints),
        cell=RawCell(scale=1.0, lattice_vectors=np.eye(3)),
    )


def test_read(raw_kpoints, Assert):
    actual = Kpoints(raw_kpoints).read()
    assert actual["mode"] == raw_kpoints.mode
    assert actual["line_length"] == raw_kpoints.number
    Assert.allclose(actual["coordinates"], raw_kpoints.coordinates)
    Assert.allclose(actual["weights"], raw_kpoints.weights)
    assert actual["labels"] is None


def test_from_file(raw_kpoints, mock_file, check_read):
    with mock_file("kpoints", raw_kpoints) as mocks:
        check_read(Kpoints, mocks, raw_kpoints)


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
        with pytest.raises(exception.RefinementError):
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
    raw_kpoints.labels = ["K", "X", " W ", "L", r"$\Gamma$"]
    raw_kpoints.label_indices = [1, 5, 6, 7, 8]
    actual = Kpoints(raw_kpoints).labels()
    ref = [""] * len(raw_kpoints.coordinates)
    ref[0] = "K"
    ref[2 * raw_kpoints.number] = "X"
    ref[3 * raw_kpoints.number - 1] = "W"
    ref[3 * raw_kpoints.number] = "L"
    ref[4 * raw_kpoints.number - 1] = r"$\Gamma$"
    assert actual == ref


def test_labels_without_data(raw_kpoints):
    actual = Kpoints(raw_kpoints).labels()
    assert actual is None
    set_line_mode(raw_kpoints)
    actual = Kpoints(raw_kpoints).labels()
    ref = [""] * len(raw_kpoints.coordinates)
    ref[0] = r"$[\frac{3}{8} \frac{3}{8} \frac{3}{4}]$"
    ref[4] = r"$[0 0 0]$"
    ref[5] = r"$[0 0 0]$"
    ref[9] = r"$[\frac{1}{2} 0 \frac{1}{2}]$"
    ref[10] = r"$[\frac{1}{2} 0 \frac{1}{2}]$"
    ref[14] = r"$[\frac{1}{2} \frac{1}{4} \frac{3}{4}]$"
    ref[15] = r"$[\frac{1}{2} \frac{1}{2} \frac{1}{2}]$"
    ref[19] = r"$[0 0 0]$"
    assert actual == ref


def test_distances_nontrivial_cell(raw_kpoints, Assert):
    cell = RawCell(
        scale=2.0,
        lattice_vectors=np.array([[3, 0, 0], [-1, 2, 0], [0, 0, 4]]),
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
    K = [0.375, 0.375, 0.75]
    GM = [0, 0, 0]
    X = [0.5, 0, 0.5]
    W = [0.5, 0.25, 0.75]
    L = [0.5, 0.5, 0.5]
    kpoints.coordinates = np.concatenate(
        (
            np.linspace(K, GM, kpoints.number),
            np.linspace(GM, X, kpoints.number),
            np.linspace(X, W, kpoints.number),
            np.linspace(L, GM, kpoints.number),
        )
    )


def test_print(raw_kpoints, format_):
    actual, _ = format_(Kpoints(raw_kpoints))
    reference = """
k-points
20
reciprocal
0 1 2  0
3 4 5  1
6 7 8  2
9 10 11  3
12 13 14  4
15 16 17  5
18 19 20  6
21 22 23  7
24 25 26  8
27 28 29  9
30 31 32  10
33 34 35  11
36 37 38  12
39 40 41  13
42 43 44  14
45 46 47  15
48 49 50  16
51 52 53  17
54 55 56  18
57 58 59  19
    """.strip()
    assert actual == {"text/plain": reference}


def test_descriptor(raw_kpoints, check_descriptors):
    kpoints = Kpoints(raw_kpoints)
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_line_length": ["line_length"],
        "_number_lines": ["number_lines"],
        "_distances": ["distances"],
        "_mode": ["mode"],
        "_labels": ["labels"],
    }
    check_descriptors(kpoints, descriptors)
