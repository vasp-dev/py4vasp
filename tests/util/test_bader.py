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


def make_structure(lattice_vectors, positions, elements):
    """Build a StructureHandler with the given geometry."""
    groups = [(name, len(list(g))) for name, g in itertools.groupby(elements)]
    raw_structure = raw.Structure(
        raw.Stoichiometry(
            number_ion_types=[count for _, count in groups],
            ion_types=[name for name, _ in groups],
        ),
        raw.Cell(lattice_vectors=np.asarray(lattice_vectors), scale=raw.VaspData(1.0)),
        positions=np.asarray(positions, dtype=float),
    )
    return StructureHandler.from_data(raw_structure)


def make_bader(lattice_vectors, positions, elements, density, snap_to_atoms=True):
    """Build a BaderAnalysis wrapping a structure with the given geometry."""
    structure = make_structure(lattice_vectors, positions, elements)
    return bader.BaderAnalysis(structure, density, snap_to_atoms=snap_to_atoms)


class ClosableStructure:
    """Wrap a structure handler to emulate an HDF5 source that closes.

    Any data access after ``closed`` is set raises, mimicking reading a lazily
    loaded VASP quantity after its file context has been left.
    """

    def __init__(self, handler):
        self._handler = handler
        self.closed = False

    def _access(self, name, *args, **kwargs):
        if self.closed:
            raise RuntimeError("Unable to synchronously read data (file closed)")
        return getattr(self._handler, name)(*args, **kwargs)

    def lattice_vectors(self):
        return self._access("lattice_vectors")

    def positions(self):
        return self._access("positions")

    def volume(self):
        return self._access("volume")

    def to_dict(self, *args, **kwargs):
        return self._access("to_dict", *args, **kwargs)

    def to_view(self, *args, **kwargs):
        return self._access("to_view", *args, **kwargs)


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
    bader_ = make_bader(lattice_vectors, [(0.5, 0.5, 0.5)], ["H"], charge, False)

    labels = bader_.basins()

    assert labels.shape == shape
    assert np.issubdtype(labels.dtype, np.integer)
    assert np.array_equal(np.unique(labels), [0])


def test_two_peaks_split_into_two_basins():
    shape = (24, 12, 12)
    lattice_vectors = np.diag((8.0, 4.0, 4.0))
    left, right = (0.25, 0.5, 0.5), (0.75, 0.5, 0.5)
    charge = gaussian_density(shape, lattice_vectors, [left, right])
    bader_ = make_bader(lattice_vectors, [(0.0, 0.0, 0.0)], ["H"], charge, False)

    labels = bader_.basins()

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
    bader_ = make_bader(lattice_vectors, atoms, ["Na", "Cl"], charge)

    labels = bader_.basins()

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

    # without snapping the interstitial peak is a basin of its own
    no_snap = make_bader(lattice_vectors, atoms, ["Na", "Cl"], charge, False)
    assert len(np.unique(no_snap.basins())) == 3

    labels = make_bader(lattice_vectors, atoms, ["Na", "Cl"], charge).basins()

    assert set(np.unique(labels)) == {0, 1}
    # the interstitial peak at 0.45 is closest to atom 0 at 0.2
    assert label_at(labels, (0.45, 0.5, 0.5)) == 0


def test_incorrect_charge_dimension_raises():
    with pytest.raises(exception.IncorrectUsage):
        make_bader(np.diag((1.0, 1.0, 1.0)), [(0.0, 0.0, 0.0)], ["H"], np.zeros((4, 4)))


@pytest.fixture
def two_atom_bader():
    shape = (24, 12, 12)
    lattice_vectors = np.diag((8.0, 4.0, 4.0))
    atoms = [(0.25, 0.5, 0.5), (0.75, 0.5, 0.5)]
    charge = gaussian_density(shape, lattice_vectors, atoms)
    bader_ = make_bader(lattice_vectors, atoms, ["Na", "Cl"], charge)
    return SimpleNamespace(bader=bader_, charge=charge, shape=shape)


def test_to_view_sets_grid_domains(two_atom_bader, Assert):
    bader_, shape = two_atom_bader.bader, two_atom_bader.shape

    view = bader_.to_view()

    assert len(view.grid_domains) == 1
    domain = view.grid_domains[0]
    assert domain.quantity.shape == (1, *shape)
    assert np.issubdtype(domain.quantity.dtype, np.integer)
    assert list(domain.labels) == ["Na_1", "Cl_1"]
    # with the default threshold nothing is hidden and ids are the snapped
    # basins shifted to 1..n_atoms
    expected = bader_.basins() + 1
    Assert.allclose(domain.quantity[0], expected)
    assert set(np.unique(domain.quantity)) == {1, 2}


def test_to_view_threshold_hides_low_density(two_atom_bader):
    bader_, charge = two_atom_bader.bader, two_atom_bader.charge
    threshold = 0.5 * charge.max()

    domain = bader_.to_view(threshold=threshold).grid_domains[0]

    hidden = charge < threshold
    assert np.all(domain.quantity[0][hidden] == 0)
    assert np.all(domain.quantity[0][~hidden] >= 1)


def test_charges_sum_to_total_integral(two_atom_bader, Assert):
    bader_, charge = two_atom_bader.bader, two_atom_bader.charge
    charges = bader_.charges()
    # VASP convention: sum(density) / density.size is the total electron count
    total = charge.sum() / charge.size
    Assert.allclose(sum(charges.values()), total)


def test_charges_keyed_by_atom_names_and_symmetric(two_atom_bader, Assert):
    charges = two_atom_bader.bader.charges()
    assert list(charges) == ["Na_1", "Cl_1"]
    # the two peaks are identical and symmetric, so the charges match
    Assert.allclose(charges["Na_1"], charges["Cl_1"])


def test_charges_of_different_density_in_same_basins(two_atom_bader, Assert):
    bader_, shape = two_atom_bader.bader, two_atom_bader.shape
    charges = bader_.charges(np.ones(shape))
    # a uniform density of one integrates to one electron over the whole grid
    Assert.allclose(sum(charges.values()), 1.0)


def expected_bader_string(charges):
    width = max(len(name) for name in charges)
    lines = [f"    {name:<{width}}   {charge:.4f}" for name, charge in charges.items()]
    return "Bader charges:\n" + "\n".join(lines)


def test_str_matches_formatted_charges(two_atom_bader):
    analysis = two_atom_bader.bader
    assert str(analysis) == expected_bader_string(analysis.charges())


def test_str_single_atom_exact():
    shape = (4, 4, 4)
    lattice_vectors = np.diag((4.0, 4.0, 4.0))
    analysis = make_bader(lattice_vectors, [(0.0, 0.0, 0.0)], ["H"], np.ones(shape))
    assert str(analysis) == "Bader charges:\n    H_1   1.0000"


def test_charges_shape_mismatch_raises(two_atom_bader):
    with pytest.raises(exception.IncorrectUsage):
        two_atom_bader.bader.charges(np.ones((2, 2, 2)))


def test_charges_non_3d_raises(two_atom_bader):
    with pytest.raises(exception.IncorrectUsage):
        two_atom_bader.bader.charges(np.ones((4, 4)))


def test_analysis_reads_no_structure_data_after_construction():
    # bader_analysis() is returned to the user after the HDF5 file context closed,
    # so the analysis must capture everything it needs at construction time.
    shape = (24, 12, 12)
    lattice_vectors = np.diag((8.0, 4.0, 4.0))
    atoms = [(0.25, 0.5, 0.5), (0.75, 0.5, 0.5)]
    charge = gaussian_density(shape, lattice_vectors, atoms)
    structure = ClosableStructure(make_structure(lattice_vectors, atoms, ["Na", "Cl"]))

    analysis = bader.BaderAnalysis(structure, charge)
    structure.closed = True  # emulate leaving the file context

    assert list(analysis.charges()) == ["Na_1", "Cl_1"]
    assert analysis.basins().shape == shape
    assert len(analysis.to_view().grid_domains) == 1


def test_plot_is_alias_of_to_view(Assert):
    shape = (20, 10, 10)
    lattice_vectors = np.diag((5.0, 4.0, 4.0))
    atoms = [(0.5, 0.5, 0.5)]
    charge = gaussian_density(shape, lattice_vectors, atoms)
    bader_ = make_bader(lattice_vectors, atoms, ["H"], charge)

    Assert.allclose(
        bader_.plot().grid_domains[0].quantity,
        bader_.to_view().grid_domains[0].quantity,
    )
