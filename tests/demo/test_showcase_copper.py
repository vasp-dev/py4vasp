# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp._demo import showcase
from py4vasp._demo.showcase import band, cell, dos, electronic_structure, structure


@pytest.fixture
def raw_dos():
    return dos.Cu("with_projectors")


@pytest.fixture
def raw_band():
    return band.Cu("with_projectors")


def test_copper_is_a_single_atom_in_a_face_centred_cell(Assert):
    raw_structure = structure.Cu()
    positions = np.array(raw_structure.positions)
    assert positions.shape == (showcase.NUMBER_STEPS, 1, 3)
    Assert.allclose(positions, np.zeros_like(positions))
    lattice_vectors = np.array(cell.Cu().lattice_vectors)
    expected = (
        cell.CU_LATTICE_CONSTANT / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    )
    Assert.allclose(lattice_vectors[-1], expected)


def test_copper_lattice_constant_matches_the_literature():
    # face-centred cubic copper, a = 3.615 Angstrom
    assert cell.CU_LATTICE_CONSTANT == pytest.approx(3.615)


def test_copper_is_face_centred_cubic():
    spglib = pytest.importorskip("spglib")
    from py4vasp._calculation.structure import Structure
    from py4vasp._calculation.symmetry import _SYMPREC

    final = Structure.from_data(structure.Cu())
    cell_tuple = (final.lattice_vectors(), final.positions(), [29])
    assert spglib.get_spacegroup(cell_tuple, symprec=_SYMPREC) == "Fm-3m (225)"


def test_metal_has_states_at_the_fermi_energy(raw_dos):
    energies, dos_values = np.array(raw_dos.energies), np.array(raw_dos.dos)[0]
    at_fermi = np.argmin(np.abs(energies - raw_dos.fermi_energy))
    # what distinguishes a metal from the other showcase systems
    assert dos_values[at_fermi] > 0.05 * dos_values.max()


def test_d_band_sits_below_the_fermi_energy(raw_dos):
    energies, dos_values = np.array(raw_dos.energies), np.array(raw_dos.dos)[0]
    peak = energies[np.argmax(dos_values)]
    # the filled d band of copper lies two to four electronvolts below the Fermi energy
    assert -4 < peak - raw_dos.fermi_energy < -2


def test_d_band_dominates_the_peak(raw_dos):
    energies = np.array(raw_dos.energies)
    projections = np.array(raw_dos.projections)[0]
    at_peak = np.argmax(np.array(raw_dos.dos)[0])
    d_orbital = projections[0, 2, at_peak]
    assert d_orbital / np.sum(projections[:, :, at_peak]) > 0.7


def test_a_band_crosses_the_fermi_energy(raw_band):
    eigenvalues = np.array(raw_band.dispersion.eigenvalues)[0]
    occupations = np.array(raw_band.occupations)[0]
    filling = occupations.sum(axis=0) / len(occupations)
    # the free-electron band is filled at some k points and empty at others
    assert np.any((filling > 0.01) & (filling < 0.99))
    assert np.min(eigenvalues) < raw_band.fermi_energy < np.max(eigenvalues)
