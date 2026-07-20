# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp._util import bader


def flat_index(shape, *multi_index):
    return np.ravel_multi_index(multi_index, shape)


def test_local_maximum_points_to_itself(Assert):
    # single peak in the corner of a 3x3x3 grid; with periodic boundaries every
    # other point is a direct (possibly diagonal) neighbor of the peak.
    charge = np.zeros((3, 3, 3))
    charge[0, 0, 0] = 1.0
    lattice_vectors = np.diag((3.0, 3.0, 3.0))

    pointers = bader._ascent_pointers(charge, lattice_vectors)

    peak = flat_index(charge.shape, 0, 0, 0)
    assert pointers[peak] == peak
    # a point on the opposite corner reaches the peak by wrapping around
    corner = flat_index(charge.shape, 2, 2, 2)
    assert pointers[corner] == peak
    # every non-peak point ascends to the single maximum
    expected = np.full(charge.size, peak)
    expected[peak] = peak
    Assert.allclose(pointers, expected)


def test_metric_weights_gradient_by_distance():
    # anisotropic cell: a step along a is much longer than a step along b, so an
    # equal density difference is a steeper ascent along b.
    charge = np.zeros((3, 3, 3))
    charge[2, 1, 1] = 1.0  # +a neighbor of (1, 1, 1)
    charge[1, 2, 1] = 1.0  # +b neighbor of (1, 1, 1)
    lattice_vectors = np.diag((30.0, 3.0, 3.0))

    pointers = bader._ascent_pointers(charge, lattice_vectors)

    source = flat_index(charge.shape, 1, 1, 1)
    steeper_neighbor = flat_index(charge.shape, 1, 2, 1)
    assert pointers[source] == steeper_neighbor
