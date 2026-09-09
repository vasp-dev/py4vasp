# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo import showcase
from py4vasp._demo.showcase import electronic_structure

ENERGY_MAXIMUM = 12.0  # eV, the whole absorption edge fits on the axis

# Lorentz oscillators as (position above the band gap, strength, width) in eV. Absorption
# cannot start below the gap, so the lowest oscillator sits at it. Sr2TiO4 is layered, so
# light polarized in the plane sees stronger and lower-lying transitions than light
# polarized along c, which is what makes the xx and zz components differ.
_IN_PLANE_OSCILLATORS = ((0.0, 9.0, 0.7), (1.9, 6.0, 1.0), (3.9, 3.5, 1.4))
_OUT_OF_PLANE_OSCILLATORS = ((0.9, 4.5, 0.8), (3.0, 6.5, 1.1), (5.2, 2.5, 1.6))
# The current-current response is a different approximation of the same spectrum, so its
# oscillators are a little weaker rather than unrelated.
_CURRENT_CURRENT_STRENGTH = 0.92


def electron() -> raw.DielectricFunction:
    """Dielectric function of Sr2TiO4 as a sum of Lorentz oscillators.

    The absorption edge starts at the gap of the showcase band structure, so the optical
    properties and the electronic structure tell the same story. The tensor is diagonal
    with equal in-plane components, as the tetragonal symmetry of the crystal requires.
    """
    energies = np.linspace(0, ENERGY_MAXIMUM, showcase.NUMBER_POINTS)
    epsilon = _tensor(energies, strength=1.0)
    current_current = _tensor(energies, strength=_CURRENT_CURRENT_STRENGTH)
    return raw.DielectricFunction(
        energies=_demo.wrap_data(energies),
        dielectric_function=_demo.wrap_data(_split_complex(epsilon)),
        current_current=_demo.wrap_data(_split_complex(current_current)),
    )


def _tensor(energies, strength):
    epsilon = np.zeros((3, 3, len(energies)), dtype=np.complex128)
    epsilon[0, 0] = epsilon[1, 1] = _lorentz(energies, _IN_PLANE_OSCILLATORS, strength)
    epsilon[2, 2] = _lorentz(energies, _OUT_OF_PLANE_OSCILLATORS, strength)
    return epsilon


def _lorentz(energies, oscillators, strength):
    # epsilon(w) = 1 + sum_j f_j / (w_j^2 - w^2 - i gamma_j w), which is causal by
    # construction: the imaginary part is positive at every energy, so the material
    # absorbs light rather than amplifying it
    epsilon = np.ones_like(energies, dtype=np.complex128)
    for offset, oscillator_strength, width in oscillators:
        position = electronic_structure.BAND_GAP + offset
        denominator = position**2 - energies**2 - 1j * width * energies
        epsilon += strength * oscillator_strength / denominator
    return epsilon


def _split_complex(epsilon):
    """Store the real and imaginary part along a trailing axis, as VASP does."""
    return np.stack((epsilon.real, epsilon.imag), axis=-1)
