# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp._demo.showcase import cell, structure

# Literature values for graphite: a = 2.4612 A, c = 6.7079 A, so the layers sit
# 3.35395 A apart and the carbon atoms 1.42104 A from each other in a layer.
LATTICE_CONSTANT = 2.4612
INTERLAYER = 3.35395
BOND_LENGTH = LATTICE_CONSTANT / np.sqrt(3)
NUMBER_LAYERS = 4
NUMBER_ATOMS = 2 * NUMBER_LAYERS


@pytest.fixture
def raw_structure():
    return structure.Graphite()


@pytest.fixture
def cartesian(raw_structure):
    positions = np.array(raw_structure.positions)
    lattice_vectors = np.array(raw_structure.cell.lattice_vectors)
    return positions @ lattice_vectors


def test_slab_holds_four_layers_of_carbon(raw_structure):
    assert np.array(raw_structure.positions).shape == (NUMBER_ATOMS, 3)
    stoichiometry = raw_structure.stoichiometry
    assert list(np.array(stoichiometry.number_ion_types)) == [NUMBER_ATOMS]
    assert [name.decode().strip() for name in np.array(stoichiometry.ion_types)] == ["C"]


def test_in_plane_cell_is_the_literature_one(raw_structure, Assert):
    lattice_vectors = np.array(raw_structure.cell.lattice_vectors)
    lengths = np.linalg.norm(lattice_vectors[:2], axis=1)
    Assert.allclose(lengths, np.full(2, LATTICE_CONSTANT))
    angle = np.degrees(
        np.arccos(np.dot(*lattice_vectors[:2]) / np.prod(lengths))
    )
    Assert.allclose(angle, 120.0)


def test_layers_sit_at_the_literature_distance(cartesian, Assert):
    heights = np.unique(np.round(cartesian[:, 2], 8))
    assert len(heights) == NUMBER_LAYERS
    Assert.allclose(np.diff(heights), np.full(NUMBER_LAYERS - 1, INTERLAYER))


def test_carbon_atoms_are_a_honeycomb(cartesian, Assert):
    lattice_vectors = np.array(structure.Graphite().cell.lattice_vectors)
    lowest = cartesian[np.isclose(cartesian[:, 2], np.min(cartesian[:, 2]))]
    shifts = np.array([[i, j, 0] for i in (-1, 0, 1) for j in (-1, 0, 1)])
    images = lowest[:, np.newaxis, :] + shifts @ lattice_vectors
    distances = np.linalg.norm(images[0] - images[1][:, np.newaxis], axis=-1)
    Assert.allclose(np.min(distances), BOND_LENGTH)


def test_layers_are_stacked_the_way_graphite_is(raw_structure, Assert):
    # Bernal stacking: half the atoms of a layer sit above an atom of the layer below
    # and half above the centre of a hexagon, which is what distinguishes graphite
    # from a stack of aligned graphene sheets
    positions = np.array(raw_structure.positions)
    layers = positions.reshape(NUMBER_LAYERS, 2, 3)[:, :, :2]
    for lower, upper in zip(layers, layers[1:]):
        eclipsed = np.isclose(lower[:, np.newaxis], upper).all(axis=-1)
        assert np.count_nonzero(eclipsed) == 1
    Assert.allclose(layers[0], layers[2])
    Assert.allclose(layers[1], layers[3])


def test_vacuum_is_wide_enough_for_a_scanning_tip(cartesian, raw_structure):
    height = np.array(raw_structure.cell.lattice_vectors)[2, 2]
    thickness = np.ptp(cartesian[:, 2])
    # py4vasp refuses to simulate a scanning tunneling microscope with less than this
    assert height - thickness > 5.0


def test_vacuum_lies_along_the_third_lattice_vector(raw_structure):
    # py4vasp checks that the atoms span less of the third direction than of the other
    # two before it places a tip above the surface
    positions = np.array(raw_structure.positions)
    span = np.ptp(positions, axis=0)
    assert span[2] < span[0] and span[2] < span[1]


def test_slab_is_centred_in_the_cell(cartesian, raw_structure, Assert):
    # an equal amount of vacuum on either side keeps the two surfaces equivalent, which
    # is what makes a single work function meaningful
    height = np.array(raw_structure.cell.lattice_vectors)[2, 2]
    below = np.min(cartesian[:, 2])
    above = height - np.max(cartesian[:, 2])
    Assert.allclose(below, above)


def test_cell_has_no_extra_scale(raw_structure):
    assert np.array(raw_structure.cell.scale) == 1.0
