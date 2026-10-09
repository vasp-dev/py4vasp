# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception, raw
from py4vasp._calculation.reaction_path import ReactionPath, ReactionPathHandler

Path = ReactionPathHandler.Path


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
    return Path(labels, atom_pairs, coordinates)


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
        Path(["a", "b", "c"], atom_pairs, coordinates)


def test_add_concatenates_in_order(path, Assert):
    other = Path(path.labels, path.atom_pairs, [[2.6, 1.15, 0.98]])
    joined = path.reversed() + other
    expected = np.concatenate([path.coordinates[::-1], other.coordinates])
    Assert.allclose(joined.coordinates, expected)
    assert joined.labels == path.labels
    Assert.allclose(joined.atom_pairs, path.atom_pairs)
    assert joined.lambda_ is None


def test_add_does_not_keep_lambda(path):
    discretized = Path(path.labels, path.atom_pairs, path.coordinates, 50.0)
    assert (discretized + discretized).lambda_ is None


@pytest.mark.parametrize(
    "atom_pairs", [[[1, 2], [1, 3], [3, 2]], [[1, 2], [1, 3], [2, 4]]]
)
def test_add_different_pairs_raises(path, atom_pairs):
    other = Path(path.labels, atom_pairs, path.coordinates)
    with pytest.raises(exception.IncorrectUsage):
        path + other


def test_add_fewer_pairs_raises(path):
    other = Path(["C~H"], [[1, 2]], [[1.0]])
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
    return Path(["x", "y"], [[1, 2], [1, 3]], np.c_[x, 2 * x])


@pytest.fixture
def curve():
    # points along a half circle, crowded at the start of the path
    angle = np.pi * np.linspace(0, 1, 200) ** 2
    coordinates = np.c_[np.cos(angle), np.sin(angle)]
    return Path(["x", "y"], [[1, 2], [1, 3]], coordinates)


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
    coarse = Path(["x"], [[1, 2]], [[0.0], [1.0], [3.0]])
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


def test_to_ICONST_lines(path):
    discretized = Path(path.labels, path.atom_pairs, path.coordinates, 49.75)
    expected = """\
R 1 2 0
R 1 3 0
R 2 3 0
IS 49.75 49.75 49.75 0
"""
    assert discretized.to_ICONST() == expected


def test_to_ICONST_without_lambda_raises(path):
    with pytest.raises(exception.IncorrectUsage):
        path.to_ICONST()


def _brute_force_distance(raw_structure, first, second):
    # minimum over all neighboring images, evaluated step by step
    lattice = raw_structure.cell.scale[()] * raw_structure.cell.lattice_vectors
    positions = raw_structure.positions
    images = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    distances = []
    for step in range(len(positions)):
        difference = positions[step, second - 1] - positions[step, first - 1]
        difference -= np.rint(difference)
        cartesian = (difference + images) @ lattice[step]
        distances.append(np.min(np.linalg.norm(cartesian, axis=1)))
    return np.array(distances)


def _hcn_trajectory():
    # H moves from C to N; atoms near the cell boundary test the minimum image
    lattice = np.array(2 * [10.0 * np.eye(3)])
    positions = np.array(
        [
            [[0.05, 0.0, 0.0], [0.95, 0.0, 0.0], [0.17, 0.0, 0.0]],
            [[0.05, 0.0, 0.0], [0.11, 0.05, 0.0], [0.17, 0.0, 0.0]],
        ]
    )
    return raw.Structure(
        raw.Stoichiometry(
            number_ion_types=np.array([1, 1, 1]), ion_types=["C", "H", "N"]
        ),
        raw.Cell(lattice_vectors=lattice, scale=raw.VaspData(1.0)),
        positions=positions,
    )


def test_read_distances_per_step(raw_data, Assert):
    structure = raw_data.structure("Sr2TiO4")
    result = ReactionPath.from_data(structure).read("3~4, 1~2")
    assert list(result) == ["3~4", "1~2"]
    Assert.allclose(result["3~4"], _brute_force_distance(structure, 3, 4))
    Assert.allclose(result["1~2"], _brute_force_distance(structure, 1, 2))
    assert len(result["3~4"]) == len(structure.positions)


def test_read_minimum_image(Assert):
    result = ReactionPath.from_data(_hcn_trajectory()).read("1~2, 2~3")
    Assert.allclose(result["1~2"], [1.0, np.sqrt(0.36 + 0.25)])
    Assert.allclose(result["2~3"], [2.2, np.sqrt(0.36 + 0.25)])


def test_read_selected_steps(raw_data, Assert):
    structure = raw_data.structure("Sr2TiO4")
    result = ReactionPath.from_data(structure)[1:3].read("3~4")
    Assert.allclose(result["3~4"], _brute_force_distance(structure, 3, 4)[1:3])


def test_read_element_pair(Assert):
    reaction_path = ReactionPath.from_data(_hcn_trajectory())
    by_element = reaction_path.read("C~H, H~N")
    by_index = reaction_path.read("1~2, 2~3")
    Assert.allclose(by_element["C~H"], by_index["1~2"])
    Assert.allclose(by_element["H~N"], by_index["2~3"])


def test_to_dict_is_alias_of_read(raw_data, Assert):
    reaction_path = ReactionPath.from_data(raw_data.structure("Sr2TiO4"))
    from_read = reaction_path.read("3~4")
    from_dict = reaction_path.to_dict("3~4")
    Assert.allclose(from_dict["3~4"], from_read["3~4"])


def test_to_path_returns_path(Assert):
    reaction_path = ReactionPath.from_data(_hcn_trajectory())
    path = reaction_path.to_path("C~H, C~N, H~N")
    assert isinstance(path, ReactionPath.Path)
    assert path.labels == ("C~H", "C~N", "H~N")
    Assert.allclose(path.atom_pairs, np.array([[1, 2], [1, 3], [2, 3]]))
    distances = reaction_path.read("C~H, C~N, H~N")
    Assert.allclose(path.coordinates, np.array(list(distances.values())).T)
    assert path.lambda_ is None


def test_Path_is_exposed_on_quantity():
    assert ReactionPath.Path is ReactionPathHandler.Path


def test_print(raw_data, format_):
    reaction_path = ReactionPath.from_data(raw_data.structure("Sr2TiO4"))
    actual, _ = format_(reaction_path)
    expected = """\
reaction path through 4 steps of Sr2TiO4
select pairs of atoms by their index, e.g. '1~2', or by element if it occurs once"""
    assert actual == {"text/plain": expected}


def test_factory_methods_access_structure(raw_data):
    data = raw_data.structure("Sr2TiO4")
    instances = (ReactionPath.from_path(), ReactionPath.from_file("vaspout.h5"))
    calls = (
        lambda quantity: quantity.read("1~2"),
        lambda quantity: quantity.to_path("1~2"),
        lambda quantity: str(quantity),
    )
    for reaction_path in instances:
        for call in calls:
            with patch("py4vasp.raw.access") as mock_access:
                mock_access.return_value.__enter__.return_value = data
                call(reaction_path)
                mock_access.assert_called_once()
                assert mock_access.call_args.args[0] == "structure"


def test_selections_lists_every_pair_of_atoms():
    selections = ReactionPath.from_data(_hcn_trajectory()).selections()
    assert selections == ["1~2", "1~3", "2~3"]
