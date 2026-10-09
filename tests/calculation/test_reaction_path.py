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
