# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp import exception
from py4vasp._util import bader


def flat_index(shape, *multi_index):
    return np.ravel_multi_index(multi_index, shape)


def gaussian_density(shape, lattice_vectors, centers, width=1.0):
    """Sum of periodic Gaussian peaks at the given fractional centers."""
    lengths = np.linalg.norm(lattice_vectors, axis=1)
    axes = [np.arange(n) / n for n in shape]
    charge = np.zeros(shape)
    for center in centers:
        distance2 = np.zeros(shape)
        grids = np.meshgrid(*axes, indexing="ij")
        for grid, frac_center, length in zip(grids, center, lengths):
            delta = (grid - frac_center + 0.5) % 1.0 - 0.5
            distance2 = distance2 + (delta * length) ** 2
        charge = charge + np.exp(-distance2 / (2 * width**2))
    return charge


def grid_point(shape, center):
    return tuple(int(round(c * n)) % n for c, n in zip(center, shape))


def label_at(labels, center):
    return labels[grid_point(labels.shape, center)]


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


def test_single_peak_is_one_basin():
    shape = (20, 18, 16)
    lattice_vectors = np.diag((6.0, 5.0, 4.0))
    charge = gaussian_density(shape, lattice_vectors, [(0.5, 0.5, 0.5)])

    labels = bader.basins(charge, lattice_vectors)

    assert labels.shape == shape
    assert np.issubdtype(labels.dtype, np.integer)
    assert np.array_equal(np.unique(labels), [0])


def test_two_peaks_split_into_two_basins():
    shape = (24, 12, 12)
    lattice_vectors = np.diag((8.0, 4.0, 4.0))
    left, right = (0.25, 0.5, 0.5), (0.75, 0.5, 0.5)
    charge = gaussian_density(shape, lattice_vectors, [left, right])

    labels = bader.basins(charge, lattice_vectors)

    assert len(np.unique(labels)) == 2
    assert label_at(labels, left) != label_at(labels, right)
    # points nearer a peak belong to that peak's basin
    assert label_at(labels, (0.1, 0.5, 0.5)) == label_at(labels, left)
    assert label_at(labels, (0.9, 0.5, 0.5)) == label_at(labels, right)


def test_displaced_peak_assigns_to_nearest_atom():
    # the density peaks are displaced inwards from the atoms, mimicking a pseudo
    # density that is not peaked at the nuclei.
    shape = (24, 12, 12)
    lattice_vectors = np.diag((8.0, 4.0, 4.0))
    # atom 0 is on the right, atom 1 on the left, so the correct labeling differs
    # from the arbitrary enumeration of the maxima and requires the snapping.
    atoms = np.array([(0.75, 0.5, 0.5), (0.25, 0.5, 0.5)])
    peaks = [(0.30, 0.5, 0.5), (0.70, 0.5, 0.5)]
    charge = gaussian_density(shape, lattice_vectors, peaks)

    labels = bader.basins(charge, lattice_vectors, atoms)

    assert set(np.unique(labels)) <= {0, 1}
    assert label_at(labels, atoms[0]) == 0
    assert label_at(labels, atoms[1]) == 1
    assert label_at(labels, (0.1, 0.5, 0.5)) == 1
    assert label_at(labels, (0.9, 0.5, 0.5)) == 0


def test_interstitial_maximum_merges_into_nearest_atom():
    shape = (30, 12, 12)
    lattice_vectors = np.diag((9.0, 4.0, 4.0))
    atoms = np.array([(0.2, 0.5, 0.5), (0.8, 0.5, 0.5)])
    charge = gaussian_density(
        shape, lattice_vectors, list(atoms), width=0.9
    ) + 0.6 * gaussian_density(shape, lattice_vectors, [(0.45, 0.5, 0.5)], width=0.4)

    # without atoms the interstitial peak is a basin of its own
    assert len(np.unique(bader.basins(charge, lattice_vectors))) == 3

    labels = bader.basins(charge, lattice_vectors, atoms)

    assert set(np.unique(labels)) == {0, 1}
    # the interstitial peak at 0.45 is closest to atom 0 at 0.2
    assert label_at(labels, (0.45, 0.5, 0.5)) == 0


def test_incorrect_charge_dimension_raises():
    with pytest.raises(exception.IncorrectUsage):
        bader.basins(np.zeros((4, 4)), np.diag((1.0, 1.0, 1.0)))


def test_incorrect_positions_shape_raises():
    charge = np.zeros((4, 4, 4))
    lattice_vectors = np.diag((1.0, 1.0, 1.0))
    with pytest.raises(exception.IncorrectUsage):
        bader.basins(charge, lattice_vectors, np.zeros((2, 2)))
