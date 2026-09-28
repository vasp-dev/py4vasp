# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest

from py4vasp._demo import showcase
from py4vasp._demo.showcase import dielectric_function, phonon, raman
from py4vasp._util import convert

NUMBER_MODES = 21  # three per atom of Sr2TiO4
NUMBER_ACOUSTIC = 3  # one per direction the crystal can translate in


@pytest.fixture
def raw_raman():
    return raman.Sr2TiO4()


@pytest.fixture
def tensor(raw_raman):
    return convert.to_complex(np.array(raw_raman.raman_tensor))


def test_tensor_has_a_mode_two_directions_and_an_energy_axis(tensor):
    assert tensor.shape == (NUMBER_MODES, 3, 3, showcase.NUMBER_POINTS)


def test_tensor_is_complex_symmetric(tensor, Assert):
    # the Raman tensor is a derivative of the susceptibility, which is symmetric under
    # exchanging the two directions. VASP writes it that way and the invariants that
    # every spectrum is built from assume it.
    Assert.allclose(tensor, np.swapaxes(tensor, 1, 2))


def test_static_limit_is_real(tensor, Assert):
    # below the gap the crystal does not absorb, so the response has no imaginary part
    Assert.allclose(tensor[..., 0].imag, np.zeros((NUMBER_MODES, 3, 3)))


def test_tensor_becomes_complex_in_resonance(tensor):
    assert np.max(np.abs(tensor.imag)) > 0.1 * np.max(np.abs(tensor.real))


def test_frequencies_match_the_phonon_modes(raw_raman, Assert):
    # the same dynamical matrix produces both, so a demo file whose two quantities
    # disagree would describe two different crystals
    modes = convert.to_complex(np.array(phonon.mode_Sr2TiO4().frequencies))
    expected = np.abs(modes) * convert.EV_TO_CM1
    Assert.allclose(np.array(raw_raman.frequencies), expected)


def test_energies_match_the_dielectric_function(raw_raman, Assert):
    # both describe how the crystal responds to light of a given energy
    expected = np.array(dielectric_function.electron().energies)
    Assert.allclose(np.array(raw_raman.energies), expected)


def test_acoustic_modes_are_not_raman_active(tensor, Assert):
    # translating the whole crystal does not change its polarizability
    acoustic = tensor[:NUMBER_ACOUSTIC]
    Assert.allclose(acoustic, np.zeros_like(acoustic))


def test_some_optical_modes_are_inactive(tensor):
    # Sr2TiO4 is centrosymmetric, so the modes split into Raman active and inactive ones
    strength = np.max(np.abs(tensor), axis=(1, 2, 3))[NUMBER_ACOUSTIC:]
    assert np.any(strength == 0) and np.any(strength > 0)
