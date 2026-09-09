# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation import _kpoints_file


@pytest.mark.parametrize(
    "name, expected",
    [
        ("GAMMA", "Γ"),
        ("SIGMA_0", "Σ₀"),
        ("DELTA_0", "Δ₀"),
        ("LAMBDA_0", "Λ₀"),
        ("K_2", "K₂"),
        ("H_12", "H₁₂"),
        ("X", "X"),
        # seekpath only labels its special points with the Greek letters above and a
        # numeric subscript; anything else is passed on as it is
        ("UNKNOWN", "UNKNOWN"),
        ("H_a", "H_a"),
    ],
)
def test_label_to_unicode(name, expected):
    assert _kpoints_file.label_to_unicode(name) == expected


@pytest.fixture
def three_segments():
    path = types.SimpleNamespace()
    path.coordinates = np.array(
        [
            [[0, 0, 0], [0.5, 0.5, 0]],
            [[0.5, 0.5, 0], [0.5, 0.75, 0.25]],
            [[0.5, 0.75, 0.25], [0, 0, 0]],
        ]
    )
    path.labels = [("Γ", "X"), ("X", "W"), ("W", "Γ")]
    path.comment = "k points along high symmetry lines"
    path.number_points = 40
    return path


def test_line_mode(three_segments):
    expected = """\
k points along high symmetry lines
40
line mode
reciprocal
  0.00000000   0.00000000   0.00000000  Γ
  0.50000000   0.50000000   0.00000000  X

  0.50000000   0.50000000   0.00000000  X
  0.50000000   0.75000000   0.25000000  W

  0.50000000   0.75000000   0.25000000  W
  0.00000000   0.00000000   0.00000000  Γ"""
    actual = _kpoints_file.line_mode(
        three_segments.coordinates,
        three_segments.labels,
        three_segments.number_points,
        three_segments.comment,
    )
    assert actual == expected


def test_line_mode_does_not_comment_labels(three_segments):
    # Many tools hide the labels behind a comment character; then VASP does not read
    # them. VASP expects the label as the fourth field of the k-point line.
    text = _kpoints_file.line_mode(
        three_segments.coordinates,
        three_segments.labels,
        three_segments.number_points,
        three_segments.comment,
    )
    assert "!" not in text
    assert "#" not in text
    kpoint_lines = [line for line in text.splitlines()[4:] if line.strip()]
    assert [line.split()[3] for line in kpoint_lines] == list("ΓXXWWΓ")


# spglib cells (lattice vectors, direct positions, atomic numbers) of the same cubic
# lattice in its primitive and its conventional setting
_SIMPLE_CUBIC = (np.eye(3), [[0, 0, 0]], [1])
_DOUBLE_CUBIC = (np.diag([2.0, 1.0, 1.0]), [[0, 0, 0], [0.5, 0, 0]], [1, 1])
_BCC_PRIMITIVE = (
    np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]),
    [[0, 0, 0]],
    [1],
)
_BCC_CONVENTIONAL = (np.eye(3), [[0, 0, 0], [0.5, 0.5, 0.5]], [1, 1])
_FCC_PRIMITIVE = (
    np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]),
    [[0, 0, 0]],
    [1],
)
_FCC_CONVENTIONAL = (
    np.eye(3),
    [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    [1, 1, 1, 1],
)


def _rotate(cell):
    "Rotate the cell rigidly; the direct coordinates of the atoms do not change."
    lattice_vectors, positions, numbers = cell
    angle = 0.3
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    return (lattice_vectors @ rotation.T, positions, numbers)


def _cartesian(coordinates, cell):
    return np.array(coordinates) @ np.linalg.inv(cell[0]).T


def test_to_input_basis_of_same_cell(Assert):
    pytest.importorskip("spglib")
    coordinates = [[0.5, -0.5, 0.5], [0.25, 0.25, 0.25]]
    actual = _kpoints_file.to_input_basis(coordinates, _BCC_PRIMITIVE, _BCC_PRIMITIVE)
    Assert.allclose(actual, coordinates)


def test_to_input_basis_of_conventional_cell(Assert):
    pytest.importorskip("spglib")
    # the H point of the bcc lattice is (1/2, -1/2, 1/2) in the primitive basis and
    # coincides with the corner (0, 1, 0) of the conventional Brillouin zone
    actual = _kpoints_file.to_input_basis(
        [[0.5, -0.5, 0.5]], _BCC_CONVENTIONAL, _BCC_PRIMITIVE
    )
    Assert.allclose(actual, [[0, 1, 0]])


@pytest.mark.parametrize(
    "cell, primitive_cell",
    [
        (_BCC_CONVENTIONAL, _BCC_PRIMITIVE),
        (_BCC_PRIMITIVE, _BCC_CONVENTIONAL),
        (_FCC_CONVENTIONAL, _FCC_PRIMITIVE),
        (_FCC_PRIMITIVE, _FCC_CONVENTIONAL),
    ],
)
def test_to_input_basis_preserves_cartesian_kpoint(cell, primitive_cell, Assert):
    pytest.importorskip("spglib")
    coordinates = [[0.5, -0.5, 0.5], [0.25, 0.25, 0.25], [0, 0, 0]]
    actual = _kpoints_file.to_input_basis(coordinates, cell, primitive_cell)
    Assert.allclose(_cartesian(actual, cell), _cartesian(coordinates, primitive_cell))


def test_to_input_basis_is_rotation_invariant(Assert):
    pytest.importorskip("spglib")
    # seekpath returns its primitive cell in an idealized orientation that need not
    # agree with the one of the input cell; the mapping must not notice
    coordinates = [[0.5, -0.5, 0.5], [0.25, 0.25, 0.25]]
    reference = _kpoints_file.to_input_basis(
        coordinates, _BCC_CONVENTIONAL, _BCC_PRIMITIVE
    )
    actual = _kpoints_file.to_input_basis(
        coordinates, _BCC_CONVENTIONAL, _rotate(_BCC_PRIMITIVE)
    )
    Assert.allclose(actual, reference)


def test_to_input_basis_rejects_supercell():
    pytest.importorskip("spglib")
    with pytest.raises(exception.IncorrectUsage):
        _kpoints_file.to_input_basis([[0, 0, 0]], _DOUBLE_CUBIC, _SIMPLE_CUBIC)


@pytest.mark.parametrize(
    "cell, determinant",
    [
        (_SIMPLE_CUBIC, 1),
        (_BCC_CONVENTIONAL, 1),
        (_BCC_PRIMITIVE, 0.5),
        (_FCC_CONVENTIONAL, 1),
        (_FCC_PRIMITIVE, 0.25),
    ],
)
def test_transformation_matrix(cell, determinant, Assert):
    pytest.importorskip("spglib")
    # the transformation matrix maps the input cell onto the standardized conventional
    # cell; its determinant is the ratio of the two volumes
    actual = _kpoints_file.transformation_matrix(cell)
    Assert.allclose(np.linalg.det(actual), determinant)


@pytest.mark.parametrize(
    "cell",
    [
        _SIMPLE_CUBIC,
        _BCC_PRIMITIVE,
        _BCC_CONVENTIONAL,
        _FCC_PRIMITIVE,
        _FCC_CONVENTIONAL,
    ],
)
def test_conventional_reciprocal_lattice(cell, Assert):
    pytest.importorskip("spglib")
    # all these cells describe a cubic lattice with a = 1, so the reciprocal lattice of
    # their conventional cell is the unit cube no matter which setting is used
    transformation = _kpoints_file.transformation_matrix(cell)
    actual = _kpoints_file.conventional_reciprocal_lattice(cell[0], transformation)
    Assert.allclose(actual, np.eye(3))


def test_conventional_reciprocal_lattice_of_tetragonal_cell(Assert):
    pytest.importorskip("spglib")
    cell = (np.diag([2.0, 2.0, 3.0]), [[0, 0, 0]], [1])
    transformation = _kpoints_file.transformation_matrix(cell)
    actual = _kpoints_file.conventional_reciprocal_lattice(cell[0], transformation)
    Assert.allclose(actual, np.diag([0.5, 0.5, 1 / 3]))


@pytest.mark.parametrize(
    "kspacing, expected",
    [
        (1.0, [2, 2, 2]),  # 1.571 -> 2
        (0.5, [3, 3, 3]),  # 3.142 rounds down to 3, a ceiling would give 4
        (0.2, [8, 8, 8]),  # 7.854 -> 8
        (1000.0, [1, 1, 1]),  # rounds to zero, but VASP always samples one k point
    ],
)
def test_divisions_from_kspacing(kspacing, expected):
    # a cubic cell with a = 4 Å has reciprocal lattice vectors of length 2π/4 ≈ 1.571/Å
    reciprocal_lattice = np.eye(3) / 4
    actual = _kpoints_file.divisions_from_kspacing(reciprocal_lattice, kspacing)
    assert actual == expected


def test_divisions_from_kspacing_of_anisotropic_cell():
    # 4 Å along the first two directions and 12 Å along the third one
    reciprocal_lattice = np.diag([0.25, 0.25, 1 / 12])
    actual = _kpoints_file.divisions_from_kspacing(reciprocal_lattice, 0.5)
    assert actual == [3, 3, 1]


def test_generating_lattice_of_conventional_cell(Assert):
    pytest.importorskip("spglib")
    # the conventional cell needs no transformation, so the mesh simply subdivides its
    # own reciprocal lattice vectors
    transformation = _kpoints_file.transformation_matrix(_SIMPLE_CUBIC)
    actual = _kpoints_file.generating_lattice(transformation, [4, 4, 2])
    Assert.allclose(actual, np.diag([0.25, 0.25, 0.5]))


def test_generating_lattice_of_primitive_cell(Assert):
    pytest.importorskip("spglib")
    # for the primitive fcc cell every conventional reciprocal lattice vector is a half
    # sum of two primitive ones, so the rows of the transformation matrix are divided by
    # the number of divisions
    transformation = _kpoints_file.transformation_matrix(_FCC_PRIMITIVE)
    actual = _kpoints_file.generating_lattice(transformation, [4, 4, 4])
    expected = [[0, 0.125, 0.125], [0.125, 0, 0.125], [0.125, 0.125, 0]]
    Assert.allclose(actual, expected)


@pytest.mark.parametrize("cell", [_FCC_PRIMITIVE, _FCC_CONVENTIONAL])
def test_generating_lattice_is_the_same_mesh_in_every_setting(cell, Assert):
    pytest.importorskip("spglib")
    # this is the point of the generalized mesh: whichever setting of the cell the user
    # provides, the k points end up on the same mesh of the conventional cell
    transformation = _kpoints_file.transformation_matrix(cell)
    generating_lattice = _kpoints_file.generating_lattice(transformation, [4, 4, 4])
    cartesian = generating_lattice @ np.linalg.inv(cell[0]).T
    Assert.allclose(cartesian, np.eye(3) / 4)
