# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp import exception
from py4vasp._util import slicing


@pytest.mark.parametrize("cut, index", [("a", 0), ("b", 1), ("c", 2)])
def test_orthorhombic(cut, index, Assert):
    cell = np.diag((3, 4, 5))
    expected_plane = np.delete(np.delete(cell, index, axis=0), index, axis=1)
    actual_plane = slicing.plane(cell, cut)
    Assert.allclose(actual_plane, expected_plane)


@pytest.mark.parametrize("cut, indices", [("a", (0, 2)), ("b", (1, 0)), ("c", (2, 1))])
def test_unusual_orthorhombic(cut, indices, Assert):
    cell = np.roll(np.diag((3, 4, 5)), shift=1, axis=0)
    expected_plane = np.delete(np.delete(cell, indices[0], axis=0), indices[1], axis=1)
    actual_plane = slicing.plane(cell, cut)
    Assert.allclose(actual_plane, expected_plane)


@pytest.mark.parametrize("cut, index", [("a", 0), ("b", 1), ("c", 2)])
def test_nearly_orthorhombic(cut, index, Assert):
    cell = 0.01 * np.ones((3, 3)) + np.diag((3, 4, 5))
    expected_plane = np.delete(cell, index, axis=0)
    # the expected plane is in 3d and the actual plane in 2d should have same angles
    # and lengths
    approximate_plane = np.delete(np.delete(cell, index, axis=0), index, axis=1)
    actual_plane = slicing.plane(cell, cut)
    Assert.allclose(actual_plane @ actual_plane.T, expected_plane @ expected_plane.T)
    assert np.max(np.abs(approximate_plane - actual_plane)) < 0.1


def test_raise_error_if_direction_is_not_obvious():
    with pytest.raises(exception.IncorrectUsage):
        slicing.plane(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), "a")


XX = np.sqrt(1 - np.sqrt(0.75))


@pytest.mark.parametrize(
    "normal, expected_plane",
    [
        ("x", [[XX, 1 + XX], [1 + XX, XX]]),
        ("y", [[1, 1], [1 + XX, -XX]]),
        ("z", [[XX + 1, -XX], [1, 1]]),
    ],
)
def test_rotate_to_user_defined_axis(normal, expected_plane, Assert):
    cell = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    actual_plane = slicing.plane(cell, "a", normal)
    Assert.allclose(actual_plane, expected_plane)


@pytest.mark.parametrize(
    "normal, rotation",
    (("x", [[1, 0], [0, 1]]), ("y", [[-1, 0], [0, 1]]), ("z", [[0, 1], [-1, 0]])),
)
def test_normal_with_orthonormal_cell(normal, rotation, Assert):
    cell = np.diag((3, 4, 5))
    expected_plane = np.dot(np.delete(np.delete(cell, 0, axis=0), 0, axis=1), rotation)
    actual_plane = slicing.plane(cell, "a", normal)
    Assert.allclose(actual_plane, expected_plane)


@pytest.mark.parametrize("cut, diagonal", [("a", [4, 5]), ("b", [3, 5]), ("c", [3, 4])])
def test_no_rotation_orthorhombic_cell(cut, diagonal, Assert):
    cell = np.diag((3, 4, 5))
    expected_plane = np.diag(diagonal)
    actual_plane = slicing.plane(cell, cut, normal=None)
    Assert.allclose(actual_plane, expected_plane)


@pytest.mark.parametrize("cut, index", [("a", 0), ("b", 1), ("c", 2)])
def test_no_rotation_nontrivial_cell(cut, index, Assert):
    cell = np.array([[0, 1, 1.1], [0.9, 0, 1.1], [0.9, 1, 0]])
    vectors = np.delete(cell, index, axis=0)
    expected_products = vectors @ vectors.T
    actual_plane = slicing.plane(cell, cut, normal=None)
    Assert.allclose(actual_plane[0, 1], 0)
    actual_products = actual_plane @ actual_plane.T
    Assert.allclose(actual_products, expected_products)
