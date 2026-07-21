# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from py4vasp import exception, raw
from py4vasp._calculation.structure import StructureHandler
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


def make_bader(lattice_vectors, positions, elements):
    """Build a Bader instance wrapping a structure with the given geometry."""
    groups = [(name, len(list(g))) for name, g in itertools.groupby(elements)]
    raw_structure = raw.Structure(
        raw.Stoichiometry(
            number_ion_types=[count for _, count in groups],
            ion_types=[name for name, _ in groups],
        ),
        raw.Cell(lattice_vectors=np.asarray(lattice_vectors), scale=raw.VaspData(1.0)),
        positions=np.asarray(positions, dtype=float),
    )
    return bader.Bader(StructureHandler.from_data(raw_structure))


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
    bader_ = make_bader(lattice_vectors, [(0.5, 0.5, 0.5)], ["H"])

    labels = bader_.basins(charge, snap_to_atoms=False)

    assert labels.shape == shape
    assert np.issubdtype(labels.dtype, np.integer)
    assert np.array_equal(np.unique(labels), [0])


def test_two_peaks_split_into_two_basins():
    shape = (24, 12, 12)
    lattice_vectors = np.diag((8.0, 4.0, 4.0))
    left, right = (0.25, 0.5, 0.5), (0.75, 0.5, 0.5)
    charge = gaussian_density(shape, lattice_vectors, [left, right])
    bader_ = make_bader(lattice_vectors, [(0.0, 0.0, 0.0)], ["H"])

    labels = bader_.basins(charge, snap_to_atoms=False)

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
    atoms = [(0.75, 0.5, 0.5), (0.25, 0.5, 0.5)]
    peaks = [(0.30, 0.5, 0.5), (0.70, 0.5, 0.5)]
    charge = gaussian_density(shape, lattice_vectors, peaks)
    bader_ = make_bader(lattice_vectors, atoms, ["Na", "Cl"])

    labels = bader_.basins(charge, snap_to_atoms=True)

    assert set(np.unique(labels)) <= {0, 1}
    assert label_at(labels, atoms[0]) == 0
    assert label_at(labels, atoms[1]) == 1
    assert label_at(labels, (0.1, 0.5, 0.5)) == 1
    assert label_at(labels, (0.9, 0.5, 0.5)) == 0


def test_interstitial_maximum_merges_into_nearest_atom():
    shape = (30, 12, 12)
    lattice_vectors = np.diag((9.0, 4.0, 4.0))
    atoms = [(0.2, 0.5, 0.5), (0.8, 0.5, 0.5)]
    charge = gaussian_density(
        shape, lattice_vectors, atoms, width=0.9
    ) + 0.6 * gaussian_density(shape, lattice_vectors, [(0.45, 0.5, 0.5)], width=0.4)
    bader_ = make_bader(lattice_vectors, atoms, ["Na", "Cl"])

    # without snapping the interstitial peak is a basin of its own
    assert len(np.unique(bader_.basins(charge, snap_to_atoms=False))) == 3

    labels = bader_.basins(charge, snap_to_atoms=True)

    assert set(np.unique(labels)) == {0, 1}
    # the interstitial peak at 0.45 is closest to atom 0 at 0.2
    assert label_at(labels, (0.45, 0.5, 0.5)) == 0


def test_incorrect_charge_dimension_raises():
    bader_ = make_bader(np.diag((1.0, 1.0, 1.0)), [(0.0, 0.0, 0.0)], ["H"])
    with pytest.raises(exception.IncorrectUsage):
        bader_.basins(np.zeros((4, 4)), snap_to_atoms=False)


@pytest.fixture
def two_atom_bader():
    shape = (24, 12, 12)
    lattice_vectors = np.diag((8.0, 4.0, 4.0))
    atoms = [(0.25, 0.5, 0.5), (0.75, 0.5, 0.5)]
    charge = gaussian_density(shape, lattice_vectors, atoms)
    bader_ = make_bader(lattice_vectors, atoms, ["Na", "Cl"])
    return SimpleNamespace(bader=bader_, charge=charge, shape=shape)


def test_to_view_sets_grid_domains(two_atom_bader, Assert):
    bader_, charge, shape = two_atom_bader.bader, two_atom_bader.charge, two_atom_bader.shape

    view = bader_.to_view(charge)

    assert len(view.grid_domains) == 1
    domain = view.grid_domains[0]
    assert domain.quantity.shape == (1, *shape)
    assert np.issubdtype(domain.quantity.dtype, np.integer)
    assert list(domain.labels) == ["Na_1", "Cl_1"]
    # with the default threshold nothing is hidden and ids are the snapped
    # basins shifted to 1..n_atoms
    expected = bader_.basins(charge, snap_to_atoms=True) + 1
    Assert.allclose(domain.quantity[0], expected)
    assert set(np.unique(domain.quantity)) == {1, 2}


def test_to_view_threshold_hides_low_density(two_atom_bader):
    bader_, charge = two_atom_bader.bader, two_atom_bader.charge
    threshold = 0.5 * charge.max()

    domain = bader_.to_view(charge, threshold=threshold).grid_domains[0]

    hidden = charge < threshold
    assert np.all(domain.quantity[0][hidden] == 0)
    assert np.all(domain.quantity[0][~hidden] >= 1)


def test_plot_is_alias_of_to_view(Assert):
    shape = (20, 10, 10)
    lattice_vectors = np.diag((5.0, 4.0, 4.0))
    atoms = [(0.5, 0.5, 0.5)]
    charge = gaussian_density(shape, lattice_vectors, atoms)
    bader_ = make_bader(lattice_vectors, atoms, ["H"])

    from_plot = bader_.plot(charge).grid_domains[0]
    from_view = bader_.to_view(charge).grid_domains[0]

    Assert.allclose(from_plot.quantity, from_view.quantity)
