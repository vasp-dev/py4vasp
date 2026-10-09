# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp import exception
from py4vasp._calculation.reaction_path import ReactionPathHandler

ReactionPath = ReactionPathHandler.ReactionPath


@pytest.fixture
def path():
    labels = ["C~H", "C~N", "H~N"]
    atom_pairs = [[1, 2], [1, 3], [2, 3]]
    coordinates = [
        [1.07, 1.16, 2.52],
        [1.20, 1.18, 1.80],
        [1.60, 1.20, 1.07],
        [2.45, 1.16, 0.99],
    ]
    return ReactionPath(labels, atom_pairs, coordinates)


def test_path_from_arrays(path, Assert):
    assert path.labels == ("C~H", "C~N", "H~N")
    Assert.allclose(path.atom_pairs, np.array([[1, 2], [1, 3], [2, 3]]))
    assert path.coordinates.shape == (4, 3)
    Assert.allclose(path.coordinates[0], np.array([1.07, 1.16, 2.52]))
    assert path.lambda_ is None


def test_reversed(path, Assert):
    reversed_path = path.reversed()
    Assert.allclose(reversed_path.coordinates, path.coordinates[::-1])
    assert reversed_path.labels == path.labels
    Assert.allclose(reversed_path.atom_pairs, path.atom_pairs)
    Assert.allclose(path.coordinates[0], np.array([1.07, 1.16, 2.52]))


@pytest.mark.parametrize(
    "atom_pairs, coordinates",
    [
        ([[1, 2], [1, 3]], [[1.0, 2.0, 3.0]]),
        ([[1, 2], [1, 3], [2, 3]], [[1.0, 2.0]]),
        ([[1, 2], [1, 3], [2, 3]], [1.0, 2.0, 3.0]),
        ([1, 2, 3], [[1.0, 2.0, 3.0]]),
    ],
)
def test_rejects_shape_mismatch(atom_pairs, coordinates):
    with pytest.raises(exception.IncorrectUsage):
        ReactionPath(["a", "b", "c"], atom_pairs, coordinates)


def test_add_concatenates_in_order(path, Assert):
    other = ReactionPath(path.labels, path.atom_pairs, [[2.6, 1.15, 0.98]])
    joined = path.reversed() + other
    expected = np.concatenate([path.coordinates[::-1], other.coordinates])
    Assert.allclose(joined.coordinates, expected)
    assert joined.labels == path.labels
    Assert.allclose(joined.atom_pairs, path.atom_pairs)
    assert joined.lambda_ is None


def test_add_does_not_keep_lambda(path):
    discretized = ReactionPath(path.labels, path.atom_pairs, path.coordinates, 50.0)
    assert (discretized + discretized).lambda_ is None


@pytest.mark.parametrize(
    "atom_pairs", [[[1, 2], [1, 3], [3, 2]], [[1, 2], [1, 3], [2, 4]]]
)
def test_add_different_pairs_raises(path, atom_pairs):
    other = ReactionPath(path.labels, atom_pairs, path.coordinates)
    with pytest.raises(exception.IncorrectUsage):
        path + other


def test_add_fewer_pairs_raises(path):
    other = ReactionPath(["C~H"], [[1, 2]], [[1.0]])
    with pytest.raises(exception.IncorrectUsage):
        path + other


def test_add_other_type_raises(path):
    with pytest.raises(TypeError):
        path + 1.0


# Reference values in the discretize tests were obtained by running the ircprepare3.py
# script of the VASP transition-state tutorial on the same curves.


@pytest.fixture
def line():
    x = np.linspace(0, 1, 101)
    return ReactionPath(["x", "y"], [[1, 2], [1, 3]], np.c_[x, 2 * x])


@pytest.fixture
def curve():
    # points along a half circle, crowded at the start of the path
    angle = np.pi * np.linspace(0, 1, 200) ** 2
    coordinates = np.c_[np.cos(angle), np.sin(angle)]
    return ReactionPath(["x", "y"], [[1, 2], [1, 3]], coordinates)


def test_discretize_uniform_line(line, Assert):
    discretized = line.discretize(6, tolerance=1e-3)
    Assert.allclose(discretized.coordinates, line.coordinates[::20])
    assert discretized.labels == line.labels
    Assert.allclose(discretized.atom_pairs, line.atom_pairs)
    Assert.allclose(discretized.lambda_, 5.0)


def test_discretize_nonuniform_curve(curve, Assert):
    discretized = curve.discretize(5, tolerance=1e-2)
    Assert.allclose(discretized.coordinates, curve.coordinates[[0, 98, 139, 170, 196]])
    Assert.allclose(discretized.lambda_, 1.808424727155654)


def test_increment_shrinks_until_tolerance_met(curve, Assert):
    # A tight tolerance shrinks the increment far below the length of the curve
    # divided by the number of points, so the selected points stop short of its end.
    discretized = curve.discretize(5, tolerance=1e-3)
    Assert.allclose(discretized.coordinates, curve.coordinates[[0, 41, 58, 71, 82]])
    Assert.allclose(discretized.lambda_, 56.31446053316491)
    spacing = np.linalg.norm(np.diff(discretized.coordinates, axis=0), axis=1)
    assert np.ptp(spacing) < 2e-3


def test_discretize_extra_points_linear(line, Assert):
    discretized = line.discretize(6, extra_points=2, tolerance=1e-3)
    x = np.linspace(-0.4, 1.4, 10)
    Assert.allclose(discretized.coordinates, np.c_[x, 2 * x])


def test_extra_points_do_not_change_lambda(curve, Assert):
    discretized = curve.discretize(5, extra_points=2, tolerance=1e-3)
    # the IRCCAR written by ircprepare3.py, which prints 6 decimals
    expected = [
        [1.017757, -0.265921],
        [1.008879, -0.132961],
        [1.000000, 0.000000],
        [0.991121, 0.132961],
        [0.964601, 0.263714],
        [0.921097, 0.389334],
        [0.861072, 0.508483],
        [0.801047, 0.627633],
        [0.741022, 0.746783],
    ]
    assert np.allclose(discretized.coordinates, expected, atol=1e-6)
    Assert.allclose(discretized.lambda_, 56.31446053316491)


@pytest.mark.parametrize(
    "number_points, extra_points, tolerance",
    [(1, 0, 1e-3), (102, 0, 1e-3), (6, -1, 1e-3), (6, 0, 0.0), (6, 0, -1e-3)],
)
def test_discretize_raises_for_invalid_arguments(
    line, number_points, extra_points, tolerance
):
    with pytest.raises(exception.IncorrectUsage):
        line.discretize(number_points, extra_points=extra_points, tolerance=tolerance)


def test_discretize_raises_if_tolerance_cannot_be_met():
    # the steps are too coarse for any spacing to be met within the tolerance
    coarse = ReactionPath(["x"], [[1, 2]], [[0.0], [1.0], [3.0]])
    with pytest.raises(exception.IncorrectUsage):
        coarse.discretize(3, tolerance=0.1)


def test_to_IRCCAR_format(curve):
    discretized = curve.discretize(5, extra_points=2, tolerance=1e-3)
    # same text as the IRCCAR written by ircprepare3.py
    expected = """\
9
 1.017757 -0.265921
 1.008879 -0.132961
 1.000000 0.000000
 0.991121 0.132961
 0.964601 0.263714
 0.921097 0.389334
 0.861072 0.508483
 0.801047 0.627633
 0.741022 0.746783
"""
    assert discretized.to_IRCCAR() == expected
