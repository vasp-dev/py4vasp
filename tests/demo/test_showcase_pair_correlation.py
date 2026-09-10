# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp._calculation.pair_correlation import PairCorrelationHandler
from py4vasp._demo import showcase
from py4vasp._demo.showcase import cell, pair_correlation, structure

ELEMENTS = np.array(["Sr", "Sr", "Ti", "O", "O", "O", "O"])
NUMBER_ATOMS = len(ELEMENTS)
SHORTEST_BOND = 1.9639  # the equatorial Ti-O bond of the relaxed crystal in Angstrom


@pytest.fixture
def raw_pair_correlation():
    return pair_correlation.Sr2TiO4()


@pytest.fixture
def distances(raw_pair_correlation):
    return np.array(raw_pair_correlation.distances)


@pytest.fixture
def functions(raw_pair_correlation):
    """The pair-correlation functions of the relaxed step, indexed by label."""
    labels = [label for label in raw_pair_correlation.labels]
    values = np.array(raw_pair_correlation.function)[-1]
    return dict(zip(labels, values))


def test_every_label_names_a_function(raw_pair_correlation):
    shape = np.array(raw_pair_correlation.function).shape
    number_pairs = 1 + NUMBER_ATOMS * (NUMBER_ATOMS + 1) // 2 - 15  # total plus 6 pairs
    assert shape == (showcase.NUMBER_STEPS, 7, showcase.NUMBER_POINTS)
    assert raw_pair_correlation.labels[0] == "total"
    assert len(raw_pair_correlation.labels) == shape[1]


def test_distances_start_at_the_origin(distances):
    assert distances[0] == 0.0
    assert distances[-1] == pair_correlation.MAX_DISTANCE
    assert np.all(np.diff(distances) > 0)


def test_no_function_is_negative(raw_pair_correlation):
    assert np.all(np.array(raw_pair_correlation.function) >= 0.0)


def test_no_atom_pair_comes_closer_than_the_shortest_bond(functions, distances):
    for label, values in functions.items():
        occupied = distances[values > 1e-3]
        assert occupied[0] > SHORTEST_BOND - 0.5, label


def test_first_peak_of_the_total_is_the_shortest_bond(functions, distances):
    total = functions["total"]
    peak = distances[np.argmax(total[distances < 2.5])]
    assert abs(peak - SHORTEST_BOND) < 2 * np.diff(distances)[0]


def test_each_pair_starts_at_its_nearest_neighbour_distance(functions, distances):
    # Where a pair function rises is the distance of the closest such pair in the
    # crystal, smeared by the broadening. Its peak is not, because neighbouring shells
    # overlap: the two strontium shells merge into a maximum that sits between them.
    for label, expected in _nearest_neighbour_distances().items():
        values = functions[label]
        onset = distances[values > 0.01 * np.max(values)][0]
        assert -4 * pair_correlation.WIDTH < onset - expected < 0.0, label


def _nearest_neighbour_distances():
    """Shortest distance of every element pair, computed from the relaxed crystal."""
    positions = np.array(structure.Sr2TiO4().positions)[-1]
    lattice = np.array(cell.Sr2TiO4().lattice_vectors)[-1]
    distances = pair_correlation.pair_distances(positions, lattice)
    nearest = {}
    for first in ("Sr", "Ti", "O"):
        for second in ("Sr", "Ti", "O"):
            if f"{second}~{first}" in nearest:
                continue
            block = distances[np.ix_(ELEMENTS == first, ELEMENTS == second)]
            nearest[f"{first}~{second}"] = np.min(block[block > 1e-8])
    return nearest


def test_functions_are_normalized_to_the_neighbour_count(functions, distances):
    # the integral of 4 pi r^2 rho g(r) counts the neighbours within that radius, which
    # for a large enough radius is the number a uniform density would give
    positions = np.array(structure.Sr2TiO4().positions)[-1]
    lattice = np.array(cell.Sr2TiO4().lattice_vectors)[-1]
    volume = np.abs(np.linalg.det(lattice))
    density = NUMBER_ATOMS / volume
    radius = pair_correlation.MAX_DISTANCE
    integrand = 4 * np.pi * distances**2 * density * functions["total"]
    neighbours = np.trapezoid(integrand, distances)
    uniform = 4 / 3 * np.pi * radius**3 * density
    assert abs(neighbours / uniform - 1) < 0.06


def test_total_is_the_sum_over_the_element_pairs(functions, Assert):
    # every pair of atoms contributes to the total, so the element-resolved functions
    # add up to it once each is weighted by the atoms it counts
    counts = {"Sr": 2, "Ti": 1, "O": 4}
    total = np.zeros_like(functions["total"])
    for label, values in functions.items():
        if label == "total":
            continue
        first, second = label.split("~")
        weight = counts[first] * counts[second]
        total += weight * values * (1 if first == second else 2)
    Assert.allclose(total / NUMBER_ATOMS**2, functions["total"])


def test_shells_move_outward_as_the_cell_expands(raw_pair_correlation, distances):
    # the trajectory starts from a cell two percent too small, so every shell sits at a
    # shorter distance and migrates outward while the crystal relaxes
    total = np.array(raw_pair_correlation.function)[:, 0]
    window = distances < 2.5
    peaks = distances[np.argmax(total[:, window], axis=1)]
    assert np.all(np.diff(peaks) >= 0)
    assert peaks[-1] > peaks[0]


def test_first_peak_py4vasp_reports_is_the_shortest_bond(raw_pair_correlation, Assert):
    # py4vasp scans for the first maximum above a threshold to summarize the function;
    # it has to find the nearest-neighbour shell, not a bump in the tail before it
    handler = PairCorrelationHandler.from_data(raw_pair_correlation)
    model = handler.to_database()
    assert abs(model.first_peak_position - SHORTEST_BOND) < 0.05
    assert model.first_peak_height > 1.0


def test_correlation_functions_are_evaluated_once(raw_pair_correlation):
    # counting the neighbour shells of twelve steps dominates the cost of building the
    # demo data, and every example in the documentation builds it again
    first = pair_correlation.Sr2TiO4()
    assert np.array(first.function) is not np.array(raw_pair_correlation.function)
    distances, function = pair_correlation._correlation_functions()
    assert not function.flags.writeable
    assert pair_correlation._correlation_functions()[1] is function
