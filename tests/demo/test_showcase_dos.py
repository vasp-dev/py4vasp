# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp._demo import showcase
from py4vasp._demo.showcase import dos as showcase_dos
from py4vasp._demo.showcase import electronic_structure


@pytest.fixture
def model():
    return electronic_structure.Sr2TiO4()


@pytest.fixture
def raw_dos():
    return showcase_dos.Sr2TiO4("with_projectors")


def test_dos_covers_the_whole_spectrum(raw_dos, model):
    energies = np.array(raw_dos.energies)
    assert energies.shape == (showcase.NUMBER_POINTS,)
    assert np.array(raw_dos.dos).shape == (1, showcase.NUMBER_POINTS)
    assert energies[0] < model.valence_band_maximum - 5
    assert energies[-1] > model.conduction_band_minimum + 3


def test_dos_is_an_insulator_at_the_fermi_energy(raw_dos, model):
    energies, dos = np.array(raw_dos.energies), np.array(raw_dos.dos)[0]
    assert np.all(dos >= 0)
    assert raw_dos.fermi_energy == pytest.approx(model.fermi_energy)
    in_gap = np.argmin(np.abs(energies - model.fermi_energy))
    # the Fermi energy sits in the middle of the gap, far enough from either band edge
    # that only the tail of the broadening reaches it
    assert dos[in_gap] < 1e-10 * dos.max()


def test_dos_carries_states_in_both_manifolds(raw_dos, model):
    energies, dos = np.array(raw_dos.energies), np.array(raw_dos.dos)[0]
    valence = dos[energies < model.valence_band_maximum]
    conduction = dos[energies > model.conduction_band_minimum]
    assert valence.max() > 0.1
    assert conduction.max() > 0.1


def test_dos_integrates_to_the_electron_count(raw_dos, model):
    energies, dos = np.array(raw_dos.energies), np.array(raw_dos.dos)[0]
    occupied = energies < model.fermi_energy
    electrons = np.trapezoid(dos[occupied], energies[occupied])
    expected = showcase_dos.SPIN_DEGENERACY * model.number_valence_bands
    assert electrons == pytest.approx(expected, rel=1e-4)


def test_projections_add_up_to_the_total(raw_dos, Assert):
    projections = np.array(raw_dos.projections)
    assert projections.shape[:3] == (1, 7, 16)
    Assert.allclose(np.sum(projections, axis=(1, 2)), np.array(raw_dos.dos))


def test_valence_manifold_is_dominated_by_oxygen_p(raw_dos, model):
    energies = np.array(raw_dos.energies)
    projections = np.array(raw_dos.projections)[0]
    valence = energies < model.valence_band_maximum
    total = np.sum(projections[:, :, valence])
    oxygen_p = np.sum(projections[3:7, 1:4, valence])
    assert oxygen_p / total > 0.7


def test_conduction_manifold_is_dominated_by_titanium_d(raw_dos, model):
    energies = np.array(raw_dos.energies)
    projections = np.array(raw_dos.projections)[0]
    conduction = energies > model.conduction_band_minimum
    total = np.sum(projections[:, :, conduction])
    titanium_t2g = np.sum(projections[2, [4, 5, 7]][:, conduction])
    assert titanium_t2g / total > 0.7


def test_dos_without_projectors_omits_the_projections():
    raw_dos = showcase_dos.Sr2TiO4("no_projectors")
    assert raw_dos.projections.is_none()
    assert raw_dos.projectors.orbital_types.is_none()
