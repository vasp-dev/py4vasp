# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp._demo.showcase import cell, grid


@pytest.fixture(params=("Sr2TiO4", "Graphite"))
def lattice_vectors(request):
    if request.param == "Sr2TiO4":
        return cell.lattice_vectors()
    return np.array(cell.Graphite().lattice_vectors)


def test_grid_has_an_odd_number_of_points(lattice_vectors):
    # an even dimension leaves |G|^2 asymmetric under G -> -G on a cell that is not
    # orthogonal, which breaks every operator evaluated by Fourier transform
    assert all(count % 2 == 1 for count in grid.grid_for(lattice_vectors))


def test_grid_samples_every_direction_equally(lattice_vectors):
    counts = grid.grid_for(lattice_vectors)
    spacing = grid.interplanar_spacing(lattice_vectors) / counts
    assert np.all(np.abs(spacing / grid.TARGET_SPACING - 1) < 0.15)


def test_interplanar_spacing_is_not_the_length_of_the_lattice_vector():
    # the primitive cell of the body-centred lattice has three vectors of equal length
    # but samples the third direction more densely, which taking the length would miss
    lattice_vectors = cell.lattice_vectors()
    lengths = np.linalg.norm(lattice_vectors, axis=1)
    spacing = grid.interplanar_spacing(lattice_vectors)
    assert np.allclose(lengths, lengths[0])
    assert not np.allclose(spacing, spacing[0])


@pytest.fixture
def field():
    """A field with two Gaussians of different width on the graphite cell."""
    lattice_vectors = np.array(cell.Graphite().lattice_vectors)
    centers = np.array([[0.0, 0.0, 0.5], [1 / 3, 2 / 3, 0.6]])
    return {
        "lattice_vectors": lattice_vectors,
        "centers": centers,
        "weights": np.array([4.0, 6.0]),
        "widths": np.array([0.5, 0.8]),
        "grid": grid.grid_for(lattice_vectors),
    }


def evaluate(field, **changes):
    arguments = {**field, **changes}
    return grid.gaussian_sum(
        arguments["centers"],
        arguments["weights"],
        arguments["widths"],
        arguments["lattice_vectors"],
        arguments["grid"],
    )


def test_field_is_positive_everywhere(field):
    assert np.all(evaluate(field) > 0.0)


def test_field_integrates_to_the_weights(field):
    # The Gaussians are normalized, so this is exact but for the tails beyond the six
    # standard deviations the images cover, which are of the order of 1e-8 of a peak.
    volume = np.abs(np.linalg.det(field["lattice_vectors"]))
    integral = np.mean(evaluate(field)) * volume
    assert abs(integral / np.sum(field["weights"]) - 1) < 1e-9


def test_field_is_periodic_in_the_lattice(field, Assert):
    # a centre moved by a lattice translation is the same centre
    shifted = field["centers"] + np.array([[1.0, -2.0, 3.0], [0.0, 1.0, 0.0]])
    Assert.allclose(evaluate(field, centers=shifted), evaluate(field))


def test_field_is_smooth_across_the_cell_boundary(field):
    # summing over the images rather than taking the closest one is what buys this: a
    # minimum-image distance has a kink where two images are equally close, and the
    # kink shows up as a crease in a contour plot
    values = evaluate(field)
    interior = np.max(np.abs(np.diff(values, axis=0)))
    across = np.max(np.abs(values[0] - values[-1]))
    assert across <= interior


def test_field_peaks_at_its_centres(field):
    values = evaluate(field)
    counts = np.array(field["grid"])
    for center in field["centers"]:
        index = np.rint(center * counts).astype(int) % counts
        at_center = values[tuple(index)]
        for axis in range(3):
            for step in (-1, 1):
                neighbour = index.copy()
                neighbour[axis] = (neighbour[axis] + step) % counts[axis]
                assert values[tuple(neighbour)] < at_center


def test_wider_gaussian_is_flatter(field):
    narrow = evaluate(field, widths=np.array([0.4, 0.4]))
    wide = evaluate(field, widths=np.array([1.2, 1.2]))
    assert np.max(narrow) > np.max(wide)
    assert np.min(narrow) < np.min(wide)


def test_field_is_evaluated_once_per_request(field):
    first = evaluate(field)
    assert not first.flags.writeable
    assert evaluate(field) is first
