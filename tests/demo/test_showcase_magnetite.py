# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp._calculation.structure import Structure
from py4vasp._demo import showcase
from py4vasp._demo.showcase import cell, projector, structure

TETRAHEDRAL = slice(0, 2)
OCTAHEDRAL = slice(2, 6)
OXYGEN = slice(6, 14)


@pytest.fixture
def raw_structure():
    return structure.Fe3O4()


@pytest.fixture
def final(raw_structure):
    return Structure.from_data(raw_structure)


def test_magnetite_has_the_primitive_spinel_cell(raw_structure):
    positions = np.array(raw_structure.positions)
    assert positions.shape == (showcase.NUMBER_STEPS, 14, 3)
    stoichiometry = raw_structure.stoichiometry
    assert list(np.array(stoichiometry.number_ion_types)) == [6, 8]


def test_magnetite_cell_is_face_centred_cubic(Assert):
    lattice_vectors = np.array(cell.Fe3O4().lattice_vectors)
    assert lattice_vectors.shape == (showcase.NUMBER_STEPS, 3, 3)
    expected = (
        cell.FE3O4_LATTICE_CONSTANT / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    )
    Assert.allclose(lattice_vectors[-1], expected)
    # the primitive cell of the face-centred lattice holds a quarter of the cubic one
    volume = abs(np.linalg.det(lattice_vectors[-1]))
    Assert.allclose(volume, cell.FE3O4_LATTICE_CONSTANT**3 / 4)


def test_magnetite_is_a_spinel(final):
    spglib = pytest.importorskip("spglib")
    from py4vasp._calculation.symmetry import _SYMPREC

    numbers = [26] * 6 + [8] * 8
    cell_tuple = (final.lattice_vectors(), final.positions(), numbers)
    assert spglib.get_spacegroup(cell_tuple, symprec=_SYMPREC) == "Fd-3m (227)"


@pytest.mark.parametrize(
    "sites, neighbors, distance",
    [(TETRAHEDRAL, 4, 1.889), (OCTAHEDRAL, 6, 2.058)],
)
def test_iron_sites_have_their_literature_coordination(
    final, sites, neighbors, distance
):
    # the two iron sublattices of magnetite differ by their coordination, and their
    # Fe-O distances are what the literature reports for the spinel
    positions, lattice = final.positions(), final.lattice_vectors()
    for site in range(*sites.indices(14)):
        delta = positions[OXYGEN] - positions[site]
        distances = np.linalg.norm((delta - np.round(delta)) @ lattice, axis=1)
        close = distances[distances < 2.4]
        assert len(close) == neighbors
        assert np.mean(close) == pytest.approx(distance, abs=1e-3)


def test_projector_matches_the_structure(raw_structure):
    raw_projector = projector.Fe3O4(use_orbitals=True)
    assert np.array_equal(
        np.array(raw_projector.stoichiometry.number_ion_types),
        np.array(raw_structure.stoichiometry.number_ion_types),
    )
    assert raw_projector.number_spin_projections == 2
    assert [orbital.decode() for orbital in np.array(raw_projector.orbital_types)] == [
        "s",
        "p",
        "d",
        "f",
    ]
