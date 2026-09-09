# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp import _demo
from py4vasp._demo import showcase
from py4vasp._demo.showcase import cell, energy, force, stress, structure, velocity

STEPS = showcase.NUMBER_STEPS
ATOMS = 7


@pytest.fixture
def ideal_positions():
    return structure.ideal_positions()


@pytest.fixture
def raw_structure():
    return structure.Sr2TiO4()


def test_every_step_resolved_quantity_covers_the_same_trajectory():
    # Force, Stress and Velocity apply their step slice to the structure they share, so a
    # quantity with fewer steps than the structure indexes past the end of its own data
    lengths = {
        "structure": len(np.array(structure.Sr2TiO4().positions)),
        "energy": len(np.array(energy.relax().values)),
        "force": len(np.array(force.Sr2TiO4().forces)),
        "stress": len(np.array(stress.Sr2TiO4().stress)),
        "velocity": len(np.array(velocity.Sr2TiO4().velocities)),
        "cell": len(np.array(cell.Sr2TiO4().lattice_vectors)),
    }
    assert set(lengths.values()) == {STEPS}, lengths


def test_positions_relax_onto_the_ideal_ones(raw_structure, ideal_positions, Assert):
    positions = np.array(raw_structure.positions)
    assert positions.shape == (STEPS, ATOMS, 3)
    # exactly, so that the symmetry analysis still recognizes the space group
    Assert.allclose(positions[-1], ideal_positions)
    deviation = np.linalg.norm(positions - ideal_positions, axis=(1, 2))
    assert np.all(np.diff(deviation) < 0)


def test_relaxation_does_not_drift(raw_structure, ideal_positions, Assert):
    # the distortion sums to zero over the atoms, so the crystal as a whole stays put and
    # the forces of every step can balance
    displacement = np.array(raw_structure.positions) - ideal_positions
    Assert.allclose(np.sum(displacement, axis=1), np.zeros((STEPS, 3)))


def test_final_structure_is_body_centred_tetragonal(raw_structure):
    spglib = pytest.importorskip("spglib")
    from py4vasp._calculation.structure import Structure
    from py4vasp._calculation.symmetry import _SYMPREC

    final = Structure.from_data(raw_structure)
    cell_tuple = (final.lattice_vectors(), final.positions(), [38, 38, 22, 8, 8, 8, 8])
    # at the tolerance py4vasp itself analyzes symmetry with, not merely at a loose one
    assert spglib.get_spacegroup(cell_tuple, symprec=_SYMPREC) == "I4/mmm (139)"


def test_ideal_positions_describe_the_same_crystal_as_the_test_data(Assert):
    # the showcase only corrects the asymmetry of the test data, it does not move an atom
    reference = np.array(_demo.structure.Sr2TiO4().positions)[-1]
    assert np.allclose(structure.ideal_positions(), reference, atol=1e-4)


def test_cell_relaxes_onto_the_tetragonal_cell(Assert):
    lattice_vectors = np.array(cell.Sr2TiO4().lattice_vectors)
    assert lattice_vectors.shape == (STEPS, 3, 3)
    # the relaxation starts from a compressed cell, so the volume grows onto its final one
    volumes = np.abs(np.linalg.det(lattice_vectors))
    assert np.all(np.diff(volumes) > 0)
    # the final cell is body-centred tetragonal: a1.a1 - a1.a2 = a^2 and a1.a1 + a1.a2 =
    # c^2 / 2 hold exactly for the standard setting
    metric = lattice_vectors[-1] @ lattice_vectors[-1].T
    Assert.allclose(metric[0, 0] - metric[0, 1], cell.LATTICE_CONSTANT**2)
    Assert.allclose(2 * (metric[0, 0] + metric[0, 1]), cell.HEIGHT**2)


def test_cell_describes_the_same_crystal_as_the_test_data():
    reference = _demo.cell.Sr2TiO4()
    vectors = np.array(reference.lattice_vectors)[-1] * float(np.array(reference.scale))
    # the same lattice up to how it is turned in space, so the metric tensors agree; the
    # showcase only replaces the rounded vectors of the test data with exact ones
    final = np.array(cell.Sr2TiO4().lattice_vectors)[-1]
    assert np.allclose(final @ final.T, vectors @ vectors.T, atol=1e-3)


def test_energy_converges_onto_its_minimum():
    raw_energy = energy.relax()
    values = np.array(raw_energy.values)
    assert values.shape == (STEPS, 3)
    free_energy = values[:, 0]
    assert np.all(np.diff(free_energy) < 0)
    # the last step is converged to better than a meV, as a relaxation would be
    assert abs(free_energy[-1] - free_energy[-2]) < 1e-3
    # a finite smearing puts the energy without entropy above the free energy and
    # extrapolates energy(sigma->0) between the two, at every step
    free, without_entropy, sigma_0 = values.T
    assert np.all(free < sigma_0)
    assert np.all(sigma_0 < without_entropy)


def test_forces_vanish_as_the_relaxation_converges(Assert):
    forces = np.array(force.Sr2TiO4().forces)
    assert forces.shape == (STEPS, ATOMS, 3)
    # Newton's third law: the forces of every step balance
    Assert.allclose(np.sum(forces, axis=1), np.zeros((STEPS, 3)))
    magnitude = np.max(np.linalg.norm(forces, axis=2), axis=1)
    assert np.all(np.diff(magnitude) < 0)
    assert magnitude[0] > 0.1
    assert magnitude[-1] < 1e-10


def test_forces_follow_the_distortion(Assert):
    forces = np.linalg.norm(np.array(force.Sr2TiO4().forces), axis=2)
    # titanium sits on its ideal site from the start, so nothing pushes it
    Assert.allclose(forces[:, 2], np.zeros(STEPS))
    # the strontium atoms are displaced most, so they feel the largest force
    assert np.all(np.argmax(forces[:-1], axis=1) < 2)


def test_stress_relaxes_and_stays_symmetric(Assert):
    stresses = np.array(stress.Sr2TiO4().stress)
    assert stresses.shape == (STEPS, 3, 3)
    Assert.allclose(stresses, np.swapaxes(stresses, 1, 2))
    assert np.all(np.diff(np.abs(np.trace(stresses, axis1=1, axis2=2))) < 0)


def test_velocities_decay_with_the_forces(Assert):
    velocities = np.array(velocity.Sr2TiO4().velocities)
    assert velocities.shape == (STEPS, ATOMS, 3)
    Assert.allclose(np.sum(velocities, axis=1), np.zeros((STEPS, 3)))
    magnitude = np.max(np.linalg.norm(velocities, axis=2), axis=1)
    assert np.all(np.diff(magnitude) < 0)
