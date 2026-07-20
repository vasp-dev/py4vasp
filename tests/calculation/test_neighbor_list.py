# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception, raw
from py4vasp._calculation import Calculation
from py4vasp._calculation.neighbor_list import NeighborList, _replica_counts
from py4vasp._calculation.structure import StructureHandler


def test_replica_counts_cubic():
    # A cube of edge a has perpendicular width a along every direction, so the
    # number of replicas is ceil(cutoff / a) in each direction.
    lattice = 4.0 * np.eye(3)
    np.testing.assert_array_equal(_replica_counts(lattice, 4.0), [1, 1, 1])
    np.testing.assert_array_equal(_replica_counts(lattice, 5.0), [2, 2, 2])
    np.testing.assert_array_equal(_replica_counts(lattice, 9.0), [3, 3, 3])


def test_replica_counts_tilted_cell():
    # a0 and a1 enclose a small angle (~5.7°). Shearing a1 keeps the volume at 1
    # but shrinks the perpendicular width along direction 0 to 1/sqrt(1+k^2)=0.1
    # while the other two directions keep width 1.
    k = np.sqrt(99.0)
    lattice = np.array([[1.0, 0.0, 0.0], [k, 1.0, 0.0], [0.0, 0.0, 1.0]])
    counts = _replica_counts(lattice, 0.95)
    # ceil(0.95 / 0.1) = 10 replicas are needed along the tilted direction
    assert counts[0] == 10
    # the other two directions have width 1, so a single replica suffices
    assert counts[1] == 1
    assert counts[2] == 1
    # a naive |a_i|-based count would use ceil(0.95 / |a0|=1) = 1 replica along
    # direction 0 and miss neighbors; the perpendicular-width criterion must not.
    naive = int(np.ceil(0.95 / np.linalg.norm(lattice[0])))
    assert counts[0] > naive


# --- helpers -----------------------------------------------------------------


def _raw_structure(lattice, positions, ion_types, number_ion_types):
    return raw.Structure(
        raw.Stoichiometry(
            number_ion_types=np.array(number_ion_types), ion_types=ion_types
        ),
        raw.Cell(lattice_vectors=np.array(lattice, dtype=float), scale=raw.VaspData(1.0)),
        positions=np.array(positions, dtype=float),
    )


def _tilted_structure():
    """A strongly sheared cell (a0, a1 enclose a small angle) with four atoms."""
    lattice = [[3.0, 0.0, 0.0], [2.9, 0.8, 0.0], [0.1, 0.2, 3.0]]
    positions = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.25, 0.1, 0.6],
        [0.7, 0.8, 0.2],
    ]
    return _raw_structure(lattice, positions, ["Si", "C"], [2, 2])


def _brute_force_map(raw_structure, cutoff, steps=None):
    """Independent O(N^2) neighbor search over an over-generous set of images."""
    handler = StructureHandler.from_data(raw_structure, steps=steps)
    lattice = np.asarray(handler.lattice_vectors())
    home = (np.asarray(handler.positions()) % 1.0) @ lattice
    volume = np.abs(np.linalg.det(lattice))
    cross = np.cross(lattice[[1, 2, 0]], lattice[[2, 0, 1]])
    reps = np.ceil(cutoff / (volume / np.linalg.norm(cross, axis=1))).astype(int) + 1
    offsets = itertools.product(*[range(-r, r + 1) for r in reps])
    result = {}
    for offset in offsets:
        shift = np.array(offset, dtype=float) @ lattice
        for i in range(len(home)):
            for j in range(len(home)):
                vector = home[j] + shift - home[i]
                distance = np.linalg.norm(vector)
                if 0 < distance <= cutoff:
                    result[(i, j, *offset)] = (distance, vector)
    return result


def _result_to_map(result):
    """Reshape a flat neighbor-list dict into {(i, j, offset): (distance, vector)}."""
    indices = np.asarray(result["indices"])
    offsets = np.asarray(result["cell_offsets"])
    distances = np.asarray(result["distances"])
    vectors = np.asarray(result["distance_vectors"])
    map_ = {}
    for row in range(len(distances)):
        i, j = indices[row]
        key = (int(i), int(j), *(int(x) for x in offsets[row]))
        map_[key] = (distances[row], vectors[row])
    return map_


def _compare_maps(actual, expected, Assert):
    assert set(actual) == set(expected)
    for key, (distance, vector) in expected.items():
        Assert.allclose(actual[key][0], distance)
        Assert.allclose(actual[key][1], vector)


# --- default (all pairs) -----------------------------------------------------


def test_simple_cubic_neighbors(Assert):
    # a single atom in a cubic cell of edge 3 Å has exactly its six periodic
    # images within a cutoff between 3 (faces) and 3*sqrt(2)~4.24 (edges).
    structure = _raw_structure([[3, 0, 0], [0, 3, 0], [0, 0, 3]], [[0, 0, 0]], ["H"], [1])
    result = NeighborList.from_data(structure).read(cutoff=3.3)
    assert len(result["distances"]) == 6
    assert np.all(result["indices"] == 0)
    Assert.allclose(result["distances"], np.full(6, 3.0))
    expected_offsets = {
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
    }
    assert {tuple(int(x) for x in off) for off in result["cell_offsets"]} == expected_offsets


def test_read_returns_expected_fields(Assert):
    structure = _raw_structure([[3, 0, 0], [0, 3, 0], [0, 0, 3]], [[0, 0, 0]], ["H"], [1])
    result = NeighborList.from_data(structure).read(cutoff=3.3)
    assert set(result) == {"indices", "distances", "distance_vectors", "cell_offsets"}
    assert result["indices"].shape == (6, 2)
    assert result["distance_vectors"].shape == (6, 3)
    assert result["cell_offsets"].shape == (6, 3)
    assert np.issubdtype(result["indices"].dtype, np.integer)
    assert np.issubdtype(result["cell_offsets"].dtype, np.integer)
    assert np.issubdtype(result["distances"].dtype, np.floating)


@pytest.mark.parametrize(
    "name, cutoff",
    [("SrTiO3", 4.3), ("ZnS", 4.1), ("BN", 3.7), ("Sr2TiO4", 4.5)],
)
def test_read_matches_brute_force(name, cutoff, raw_data, Assert):
    structure = raw_data.structure(name)
    result = NeighborList.from_data(structure).read(cutoff=cutoff)
    _compare_maps(_result_to_map(result), _brute_force_map(structure, cutoff), Assert)


def test_read_matches_brute_force_tilted_cell(Assert):
    # robustness for a strongly sheared cell (small enclosed angle)
    structure = _tilted_structure()
    result = NeighborList.from_data(structure).read(cutoff=4.0)
    _compare_maps(_result_to_map(result), _brute_force_map(structure, 4.0), Assert)


def test_neighbor_pairs_are_symmetric():
    # every (i, j, offset) has its mirror (j, i, -offset)
    structure = _tilted_structure()
    result = NeighborList.from_data(structure).read(cutoff=4.0)
    map_ = _result_to_map(result)
    for (i, j, ox, oy, oz) in map_:
        assert (j, i, -ox, -oy, -oz) in map_


# --- selection ---------------------------------------------------------------


def _elements(structure):
    return StructureHandler.from_data(structure).to_dict()["elements"]


def _filter_map(brute_map, elements, source_type, neighbor_type=None):
    return {
        key: value
        for key, value in brute_map.items()
        if elements[key[0]] == source_type
        and (neighbor_type is None or elements[key[1]] == neighbor_type)
    }


def test_read_pair_selection(Assert):
    structure = _tilted_structure()
    elements = _elements(structure)
    result = NeighborList.from_data(structure).read("Si~C", cutoff=4.0)
    assert set(result) == {"Si~C"}
    expected = _filter_map(_brute_force_map(structure, 4.0), elements, "Si", "C")
    _compare_maps(_result_to_map(result["Si~C"]), expected, Assert)


def test_read_pair_selection_reversed(Assert):
    structure = _tilted_structure()
    elements = _elements(structure)
    result = NeighborList.from_data(structure).read("C~Si", cutoff=4.0)
    expected = _filter_map(_brute_force_map(structure, 4.0), elements, "C", "Si")
    _compare_maps(_result_to_map(result["C~Si"]), expected, Assert)


def test_read_bare_element_selection(Assert):
    structure = _tilted_structure()
    elements = _elements(structure)
    result = NeighborList.from_data(structure).read("Si", cutoff=4.0)
    expected = _filter_map(_brute_force_map(structure, 4.0), elements, "Si")
    _compare_maps(_result_to_map(result["Si"]), expected, Assert)


def test_read_multiple_selections(Assert):
    structure = _tilted_structure()
    elements = _elements(structure)
    result = NeighborList.from_data(structure).read("Si~C, C~C", cutoff=4.0)
    assert set(result) == {"Si~C", "C~C"}
    brute = _brute_force_map(structure, 4.0)
    _compare_maps(
        _result_to_map(result["Si~C"]), _filter_map(brute, elements, "Si", "C"), Assert
    )
    _compare_maps(
        _result_to_map(result["C~C"]), _filter_map(brute, elements, "C", "C"), Assert
    )


def test_read_unknown_element_raises():
    structure = _tilted_structure()
    with pytest.raises(exception.IncorrectUsage):
        NeighborList.from_data(structure).read("Xx~C", cutoff=4.0)


def test_read_unsupported_selection_raises():
    # a range (":") is not a meaningful pair selection for a neighbor list
    structure = _tilted_structure()
    with pytest.raises(exception.IncorrectUsage):
        NeighborList.from_data(structure).read("Si:C", cutoff=4.0)


# --- aliases, steps, print, factory, exposure --------------------------------


def test_to_dict_is_alias_of_read(Assert):
    structure = _tilted_structure()
    neighbor_list = NeighborList.from_data(structure)
    from_read = neighbor_list.read(cutoff=4.0)
    from_dict = neighbor_list.to_dict(cutoff=4.0)
    assert from_read.keys() == from_dict.keys()
    for key in from_read:
        Assert.allclose(from_dict[key], from_read[key])


def test_read_single_step(raw_data, Assert):
    # Sr2TiO4 demo data is a trajectory; [step] selects a single geometry.
    structure = raw_data.structure("Sr2TiO4")
    result = NeighborList.from_data(structure)[0].read(cutoff=4.5)
    expected = _brute_force_map(structure, 4.5, steps=0)
    _compare_maps(_result_to_map(result), expected, Assert)


def test_read_multiple_steps_not_implemented(raw_data):
    structure = raw_data.structure("Sr2TiO4")
    with pytest.raises(exception.NotImplemented):
        NeighborList.from_data(structure)[0:2].read(cutoff=4.5)


def test_print(raw_data, format_):
    structure = raw_data.structure("Sr2TiO4")
    neighbor_list = NeighborList.from_data(structure)
    actual, _ = format_(neighbor_list)
    assert actual == {"text/plain": "neighbor list of 7 atoms (Sr, Ti, O)"}


def test_factory_methods_access_structure(raw_data):
    # NeighborList owns no data of its own; from_path/from_file must access the
    # structure in the schema rather than a nonexistent "neighbor_list" entry.
    data = raw_data.structure("Sr2TiO4")
    instances = (NeighborList.from_path(), NeighborList.from_file("vaspout.h5"))
    calls = (lambda nl: nl.read(cutoff=4.0), lambda nl: str(nl))
    for neighbor_list in instances:
        for call in calls:
            with patch("py4vasp.raw.access") as mock_access:
                mock_access.return_value.__enter__.return_value = data
                call(neighbor_list)
                mock_access.assert_called_once()
                assert mock_access.call_args.args[0] == "structure"


def test_calculation_exposes_neighbor_list():
    calculation = Calculation.from_path(".")
    assert isinstance(calculation.neighbor_list, NeighborList)
