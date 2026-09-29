# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo.showcase import dielectric_function, electronic_structure, phonon
from py4vasp._util import convert

# Resonances of the susceptibility as (position above the band gap, strength, width) in
# eV. They make the Raman tensor real below the gap, where the crystal does not absorb,
# and complex once the laser reaches a transition -- which is the resonance a user goes
# looking for when they scan the laser energy.
_RESONANCES = ((0.4, 1.0, 0.9), (2.6, 0.55, 1.3))
# Strength of the strongest mode; the others are scaled down from it so that a spectrum
# has a few dominant lines rather than twenty-one equal ones, as a real one does.
_MAXIMUM_ACTIVITY = 4.0


def Sr2TiO4() -> raw.Raman:
    """Raman tensor of Sr2TiO4 for every zone-centre mode.

    Sr2TiO4 has an inversion centre, so its modes are either Raman active or infrared
    active but never both. The active ones here carry the tensor shapes the tetragonal
    symmetry allows, and the three acoustic modes carry none at all: translating the
    whole crystal cannot change its polarizability.
    """
    energies = np.array(dielectric_function.electron().energies)
    tensors = _tensors()[:, :, :, np.newaxis] * _resonance(energies)
    return raw.Raman(
        frequencies=_demo.wrap_data(_frequencies()),
        energies=_demo.wrap_data(energies),
        raman_tensor=_demo.wrap_data(_split_complex(tensors)),
    )


def _frequencies():
    """Frequency of every zone-centre mode in cm^-1, the unit VASP uses for Raman.

    The modes are the ones the dispersion reports at Gamma, derived the same way
    :func:`phonon.mode_Sr2TiO4` derives them, so the two quantities of the demo file
    describe one crystal rather than two.
    """
    at_gamma = phonon.branch_frequencies(np.zeros((1, 3)))[0] / convert.EV_TO_THZ
    # VASP reports a magnitude here, so an unstable mode would be indistinguishable
    # from a stable one. The showcase crystal is stable, so nothing is lost.
    return np.abs(at_gamma) * convert.EV_TO_CM1


def _resonance(energies):
    # chi(w) = sum_j f_j / (w_j^2 - w^2 - i gamma_j w), the same oscillator model the
    # showcase dielectric function uses, so both describe the same electronic structure
    profile = np.zeros_like(energies, dtype=np.complex128)
    for offset, strength, width in _RESONANCES:
        position = electronic_structure.BAND_GAP + offset
        profile += strength / (position**2 - energies**2 - 1j * width * energies)
    return profile


def _tensors():
    """Raman tensor of every mode at zero photon energy, shape ``(mode, 3, 3)``."""
    tensors = np.zeros((phonon.NUMBER_MODES, 3, 3))
    optical = tensors[phonon.NUMBER_ACOUSTIC :]
    for index, tensor in enumerate(optical):
        tensor[:] = _symmetry_allowed_tensor(index)
    # the lowest modes move the heavy sublattice and polarize it least, so the
    # oxygen modes at the top of the spectrum dominate the spectrum
    strengths = np.linspace(0.2, 1.0, len(optical)) * _MAXIMUM_ACTIVITY
    optical *= strengths[:, np.newaxis, np.newaxis]
    return tensors


def _symmetry_allowed_tensor(index):
    """One of the tensor shapes a tetragonal crystal allows, or none at all.

    A mode of a centrosymmetric crystal is Raman active only if it leaves the inversion
    centre alone, so every other mode here is silent. The active ones cycle through the
    three shapes the symmetry permits: a fully symmetric one, one that distinguishes the
    two in-plane axes, and one that shears the plane against the axis.
    """
    if index % 2:
        return np.zeros((3, 3))  # infrared active instead, by the mutual exclusion rule
    if index % 6 == 0:  # fully symmetric, the strongest lines of an oxide
        return np.diag((1.0, 1.0, 0.6))
    if index % 6 == 2:  # in-plane, traceless, so it scatters only depolarized light
        return np.diag((1.0, -1.0, 0.0))
    return np.array([[0.0, 0.0, 0.8], [0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])  # shear


def _split_complex(tensors):
    """Store the real and imaginary part along a trailing axis, as VASP does."""
    return np.stack((tensors.real, tensors.imag), axis=-1)
