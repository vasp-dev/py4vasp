# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp._calculation.symmetry import SymmetryHandler
from py4vasp._demo.showcase import cell, structure, symmetry

NUMBER_OPERATIONS = 16  # the order of the point group 4/mmm
NUMBER_ATOMS = 7


@pytest.fixture
def raw_symmetry():
    return symmetry.Sr2TiO4()


@pytest.fixture
def rotations(raw_symmetry):
    return np.array(raw_symmetry.rotations)


def test_group_has_the_order_of_its_point_group(raw_symmetry, rotations):
    assert int(raw_symmetry.number_of_operations) == NUMBER_OPERATIONS
    assert rotations.shape == (NUMBER_OPERATIONS, 3, 3)
    assert len({tuple(rotation.ravel()) for rotation in rotations}) == NUMBER_OPERATIONS


def test_group_is_symmorphic(raw_symmetry):
    # I4/mmm places every atom of Sr2TiO4 on a site the rotations alone leave invariant,
    # so no operation needs a fractional translation to accompany it
    assert np.all(np.array(raw_symmetry.translations) == 0.0)


def test_every_operation_maps_the_crystal_onto_itself(rotations):
    positions = structure.ideal_positions()
    for rotation in rotations:
        images = (positions @ rotation.T) % 1.0
        distances = _distance_to_nearest_atom(positions, images)
        assert np.max(distances) < 1e-10


def _distance_to_nearest_atom(positions, images):
    difference = positions[np.newaxis] - images[:, np.newaxis]
    difference -= np.rint(difference)
    return np.min(np.linalg.norm(difference, axis=-1), axis=-1)


def test_operations_are_the_tetragonal_point_group(rotations):
    # a point group is closed under multiplication and contains the inverse of every
    # element, which catches a table that is merely 16 plausible-looking matrices
    known = {tuple(rotation.ravel()) for rotation in rotations}
    for first in rotations:
        for second in rotations:
            assert tuple((first @ second).ravel()) in known
    assert tuple(np.eye(3, dtype=int).ravel()) in known


def test_crystal_is_centrosymmetric(rotations):
    # the inversion is what distinguishes I4/mmm from the tetragonal groups without it,
    # and Symmetry.has_inversion_symmetry reports it
    assert any(
        np.array_equal(rotation, -np.eye(3, dtype=int)) for rotation in rotations
    )


def test_inverse_operations_point_at_the_inverse(raw_symmetry, rotations):
    inverse = np.array(raw_symmetry.inverse_operations)
    for rotation, index in zip(rotations, inverse):
        product = rotation @ rotations[index - 1]  # the file stores Fortran indices
        assert np.array_equal(product, np.eye(3, dtype=int))


def test_reciprocal_rotations_transform_the_reciprocal_lattice(raw_symmetry, rotations):
    reciprocal = np.array(raw_symmetry.reciprocal_rotations)
    for rotation, image in zip(rotations, reciprocal):
        assert np.allclose(image, np.linalg.inv(rotation).T)


def test_atom_permutations_permute_every_atom(raw_symmetry):
    permutations = np.array(raw_symmetry.atom_permutations)
    assert permutations.shape == (1, NUMBER_OPERATIONS, NUMBER_ATOMS)
    for permutation in permutations[0]:
        assert sorted(permutation) == list(range(1, NUMBER_ATOMS + 1))


def test_atom_permutations_keep_the_element_of_every_atom(raw_symmetry):
    # a symmetry operation may only map an atom onto an atom of the same element
    elements = np.array(["Sr", "Sr", "Ti", "O", "O", "O", "O"])
    for permutation in np.array(raw_symmetry.atom_permutations)[0]:
        assert np.array_equal(elements[permutation - 1], elements)


def test_cell_is_the_relaxed_showcase_cell(raw_symmetry, Assert):
    # the operations belong to the ideal positions, which the trajectory relaxes onto,
    # so the symmetry describes the final cell rather than a compressed intermediate one
    lattice_vectors = np.array(raw_symmetry.cell.lattice_vectors)
    scale = np.array(raw_symmetry.cell.scale)
    Assert.allclose(scale * lattice_vectors, cell.lattice_vectors())


def test_computational_cell_is_the_primitive_one(raw_symmetry, Assert):
    assert int(raw_symmetry.number_of_primitive_cells) == 1
    Assert.allclose(
        np.array(raw_symmetry.primitive_lattice_vectors), cell.lattice_vectors()
    )
    Assert.allclose(np.array(raw_symmetry.primitive_translations), np.zeros((1, 3)))


def test_spglib_resolves_the_space_group_of_the_crystal(raw_symmetry):
    pytest.importorskip("spglib")
    handler = SymmetryHandler.from_data(raw_symmetry)
    space_group = handler.space_group()
    assert space_group.number == 139
    assert space_group.international_symbol == "I4/mmm"
    assert space_group.is_symmorphic
    assert handler.bravais_lattice() == "tI"
    # the conventional cell of a body-centred lattice holds twice the primitive one
    assert handler.pearson_symbol() == "tI14"
    assert handler.has_inversion_symmetry()
