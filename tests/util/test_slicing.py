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
