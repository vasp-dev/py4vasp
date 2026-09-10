# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp._calculation.structure import Structure
from py4vasp._demo import showcase
from py4vasp._demo.showcase import band, cell, dos, local_moment, projector, structure

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


@pytest.fixture
def raw_dos():
    return dos.Fe3O4("with_projectors")


@pytest.fixture
def raw_band():
    return band.Fe3O4("with_projectors")


def test_dos_resolves_two_spin_channels(raw_dos):
    dos_values = np.array(raw_dos.dos)
    assert dos_values.shape == (2, showcase.NUMBER_POINTS)
    # the two channels must differ everywhere, or an index bug swapping them would pass
    assert not np.allclose(dos_values[0], dos_values[1])
    assert np.all(dos_values >= 0)


def test_magnetite_is_a_half_metal(raw_dos):
    energies, dos_values = np.array(raw_dos.energies), np.array(raw_dos.dos)
    at_fermi = np.argmin(np.abs(energies - raw_dos.fermi_energy))
    majority, minority = dos_values
    # one channel is gapped at the Fermi energy and the other carries states there
    assert majority[at_fermi] < 1e-6 * majority.max()
    assert minority[at_fermi] > 0.05 * minority.max()


def test_band_has_a_partly_filled_minority_band(raw_band):
    occupations = np.array(raw_band.occupations)
    assert occupations.shape[0] == 2
    majority, minority = occupations
    # the majority channel is either full or empty per band, the minority band that
    # crosses the Fermi energy is filled at some k points and empty at others
    assert set(np.unique(majority.sum(axis=0) / len(majority))) <= {0.0, 1.0}
    filling = minority.sum(axis=0) / len(minority)
    assert np.any((filling > 0.01) & (filling < 0.99))


def test_local_moments_are_ferrimagnetic(Assert):
    raw_moment = local_moment.Fe3O4()
    moments = np.array(raw_moment.spin_moments)
    assert moments.shape == (showcase.NUMBER_STEPS, 2, 14, 3)
    total_per_atom = np.sum(moments[-1, 1], axis=-1)
    # the tetrahedral sublattice orders antiparallel to the octahedral one
    assert np.all(total_per_atom[TETRAHEDRAL] < 0)
    assert np.all(total_per_atom[OCTAHEDRAL] > 0)
    # and the two do not cancel: magnetite carries 4 Bohr magnetons per formula unit,
    # so eight for the two formula units of the primitive cell
    Assert.allclose(np.sum(total_per_atom), 8.0)


def test_charges_are_positive_and_larger_on_oxygen(Assert):
    charges = np.array(local_moment.Fe3O4().spin_moments)[-1, 0]
    assert np.all(charges >= 0)
    # oxygen keeps its 2p shell nearly full, so its p charge dominates
    assert np.all(charges[OXYGEN, 1] > charges[OXYGEN, 2])
    # iron carries its charge in the d shell
    assert np.all(charges[TETRAHEDRAL, 2] > charges[TETRAHEDRAL, 1])


@pytest.fixture
def noncollinear_dos():
    return dos.Fe3O4("with_projectors", "noncollinear")


def test_noncollinear_dos_resolves_charge_and_three_spin_components(noncollinear_dos):
    dos_values = np.array(noncollinear_dos.dos)
    assert dos_values.shape == (4, showcase.NUMBER_POINTS)
    charge, *spin = dos_values
    assert np.all(charge >= 0)
    # the three components must differ from one another, or an index bug that mixes them
    # up would go unnoticed
    for first, second in ((0, 1), (0, 2), (1, 2)):
        assert not np.allclose(spin[first], spin[second])


def test_noncollinear_charge_is_the_sum_of_both_channels(noncollinear_dos, Assert):
    collinear = np.array(dos.Fe3O4("no_projectors").dos)
    charge = np.array(noncollinear_dos.dos)[0]
    # the same material, so the charge of the noncollinear calculation is what the two
    # channels of the collinear one add up to
    Assert.allclose(charge, np.sum(collinear, axis=0))


def test_noncollinear_spin_points_along_the_sublattices(noncollinear_dos):
    energies = np.array(noncollinear_dos.energies)
    sigma_z = np.array(noncollinear_dos.dos)[3]
    # the two iron sublattices order antiparallel, so the projection along z changes sign
    # between the states the one dominates and the states the other does
    assert np.max(sigma_z) > 0
    assert np.min(sigma_z) < 0


def test_noncollinear_band_has_one_set_of_eigenvalues():
    raw_band = band.Fe3O4("with_projectors", magnetism="noncollinear")
    eigenvalues = np.array(raw_band.dispersion.eigenvalues)
    collinear = np.array(band.Fe3O4("no_projectors").dispersion.eigenvalues)
    # a noncollinear calculation does not split the bands by spin; the spin appears in
    # the four components of the projections instead. It does not lose half the bands
    # either: the spinor basis holds as many as the two channels have together.
    assert eigenvalues.shape[0] == 1
    assert eigenvalues.shape[2] == 2 * collinear.shape[2]
    assert np.array(raw_band.projections).shape == (4, 14, 4, *eigenvalues.shape[1:])


def test_noncollinear_band_agrees_with_its_density_of_states(Assert):
    # Band and Dos are written from the same models, so a gap in one and states at the
    # Fermi energy in the other would mean the two disagree about the material
    raw_band = band.Fe3O4("no_projectors", magnetism="noncollinear")
    raw_dos = dos.Fe3O4("no_projectors", "noncollinear")
    eigenvalues = np.array(raw_band.dispersion.eigenvalues)[0]
    energies, charge = np.array(raw_dos.energies), np.array(raw_dos.dos)[0]
    at_fermi = np.argmin(np.abs(energies - raw_dos.fermi_energy))
    assert charge[at_fermi] > 0.05 * charge.max()
    below = eigenvalues[eigenvalues < raw_band.fermi_energy].max()
    above = eigenvalues[eigenvalues > raw_band.fermi_energy].min()
    # the density of states carries weight at the Fermi energy, so no band may leave a
    # gap around it that the broadening could not have filled
    assert above - below < 4 * showcase.BROADENING


def test_noncollinear_local_moments_carry_all_three_axes(Assert):
    raw_moment = local_moment.Fe3O4("noncollinear")
    moments = np.array(raw_moment.spin_moments)
    assert moments.shape == (showcase.NUMBER_STEPS, 4, 14, 3)
    assert not raw_moment.orbital_moments.is_none()
    # the length of the moment of every atom is what the collinear calculation reports
    vector = np.sum(moments[-1, 1:], axis=-1)
    collinear = np.sum(np.array(local_moment.Fe3O4().spin_moments)[-1, 1], axis=-1)
    Assert.allclose(np.linalg.norm(vector, axis=0), np.abs(collinear))
