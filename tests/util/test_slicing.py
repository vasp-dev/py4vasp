# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

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
