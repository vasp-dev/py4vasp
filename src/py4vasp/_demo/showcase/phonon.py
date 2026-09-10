# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import functools

import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo import showcase
from py4vasp._demo.showcase import electronic_structure, kpoint, structure

NUMBER_ATOMS = 7
NUMBER_MODES = 3 * NUMBER_ATOMS
NUMBER_ACOUSTIC = 3  # one per direction a sound wave can travel in
SOUND_VELOCITY = 22.0  # THz per unit of the reduced q vector, the slope at Gamma
# Atomic masses of the atoms of Sr2TiO4 in the order the structure lists them. They
# decide how the displacement of a mode is shared between the atoms: VASP reports
# mass-weighted displacements, so a heavy atom carries a large share of a mode that
# moves every atom by the same amount.
MASSES = (87.62, 87.62, 47.87, 16.00, 16.00, 16.00, 16.00)
# Exponents of the softness 1/sqrt(mass) that the optical branches sweep through. The
# lowest optical modes vibrate the heavy sublattice and the highest ones are the oxygen
# stretching modes, so the share moves from the cations to the anions with frequency.
OPTICAL_EXPONENTS = (-6.0, 6.0)
BROADENING = 0.25  # THz, wide enough to sample smoothly and narrow enough to resolve
# VASP stores the eigenvalues of the dynamical matrix as energies, so the showcase
# converts its frequencies; py4vasp prints them back in THz with the same factor.
EV_TO_THZ = 241.798934781
ENERGY_MARGIN = 1.5  # THz of empty axis above the highest branch
# Frequency in THz that every optical branch reaches at the zone centre. Sr2TiO4 has
# eighteen of them, spread over the range an oxide occupies; the heavier the atoms
# involved, the lower the branch.
OPTICAL_AT_GAMMA = (
    3.2,
    3.6,
    4.1,
    4.9,
    5.4,
    6.0,
    6.7,
    7.3,
    8.1,
    9.0,
    9.8,
    10.6,
    11.5,
    12.4,
    13.6,
    15.1,
    17.2,
    19.4,
)
_OPTICAL_WIDTH = 0.55  # how much a branch disperses away from the zone centre


def band_Sr2TiO4() -> raw.PhononBand:
    """Phonon dispersion of Sr2TiO4 along the high-symmetry path.

    Three acoustic branches rise linearly out of the zone centre, as sound waves do, and
    eighteen optical branches sit above them. Every frequency is positive, which is what
    a structure at its energy minimum gives; a negative one would mark an unstable mode.
    """
    qpoints = kpoint.line_mode()
    coordinates = np.array(qpoints.coordinates)
    by_branch = branch_frequencies(coordinates)
    # VASP writes the frequencies of every q point in ascending order. The acoustic
    # branches overtake the lower optical ones away from Gamma, so the two groups have to
    # be sorted together, and the displacement patterns follow the same permutation to
    # stay with their frequency.
    frequencies, order = electronic_structure.sort_bands(by_branch)
    eigenvectors = _eigenvectors(len(coordinates))
    return raw.PhononBand(
        dispersion=raw.Dispersion(qpoints, _demo.wrap_data(frequencies)),
        stoichiometry=_demo.stoichiometry.Sr2TiO4(has_ion_types=True),
        eigenvectors=_demo.wrap_data(
            np.take_along_axis(eigenvectors, _expand(order), axis=1)
        ),
        primitive_positions=_demo.wrap_data(structure.ideal_positions()),
    )


def _expand(order):
    """Shape the permutation so it applies to the trailing axes of the eigenvectors."""
    return order[:, :, np.newaxis, np.newaxis, np.newaxis]


def _acoustic(coordinates):
    """Three branches rising linearly from zero at the zone centre."""
    # The form a chain of masses and springs gives: linear in q near Gamma, as sound
    # waves are, and flat where it meets the zone boundary. Measuring the distance in
    # the reduced coordinates instead would leave a kink at the boundary.
    distance = np.linalg.norm(np.sin(np.pi * coordinates), axis=1) / np.pi
    # the two transverse branches are slower than the longitudinal one
    velocities = np.array([0.62, 0.68, 1.0]) * SOUND_VELOCITY
    return np.multiply.outer(distance, velocities)


def _optical(coordinates):
    """Eighteen branches dispersing gently around their zone-centre frequency."""
    reduced = coordinates - np.round(coordinates)
    # each branch disperses along a different combination of the reciprocal directions,
    # so the bundle fans out instead of running in parallel
    phases = np.arange(1, len(OPTICAL_AT_GAMMA) + 1)
    modulation = np.cos(2 * np.pi * np.sum(reduced, axis=1))
    shape = (len(coordinates), len(OPTICAL_AT_GAMMA))
    dispersion = _OPTICAL_WIDTH * np.outer(modulation - 1, np.sqrt(phases) / 4)
    return np.broadcast_to(OPTICAL_AT_GAMMA, shape) + dispersion


def branch_frequencies(coordinates) -> np.ndarray:
    """Frequency of every branch at every given q point, shape ``(qpoint, branch)``.

    The branches are in the order the model defines them, the acoustic ones first, not
    sorted by frequency. A dispersion has to sort them because VASP does; a density of
    states sums over all of them, so it pairs every frequency with the displacement
    pattern of its own branch without sorting.
    """
    return np.concatenate((_acoustic(coordinates), _optical(coordinates)), axis=1)


def mode_weights() -> np.ndarray:
    """Share of every atom in the displacement of every mode, shape ``(mode, atom)``.

    Returns
    -------
    -
        Shares that sum to one per mode. The acoustic modes translate the whole cell, so
        every atom moves by the same amount and the mass-weighted share is proportional
        to the mass. The optical modes sweep from the heavy sublattice to the light one,
        which is what makes the projected density of states of an oxide informative:
        strontium carries the bottom of the spectrum and oxygen the top.
    """
    masses = np.array(MASSES)
    acoustic = np.broadcast_to(masses / np.sum(masses), (NUMBER_ACOUSTIC, NUMBER_ATOMS))
    softness = 1 / np.sqrt(masses)
    exponents = np.linspace(*OPTICAL_EXPONENTS, len(OPTICAL_AT_GAMMA))[:, np.newaxis]
    optical = softness**exponents
    return np.concatenate((acoustic, optical / np.sum(optical, axis=1, keepdims=True)))


def _eigenvectors(number_qpoints):
    """Normalized displacement patterns, one per mode and q point."""
    # a deterministic set of patterns: mode n displaces the atoms along a smooth wave
    # whose wavelength grows with n, rescaled so that each atom carries the share
    # mode_weights assigns to it and the pattern stays normalized per mode
    mode = np.arange(NUMBER_MODES)[:, np.newaxis, np.newaxis]
    atom = np.arange(NUMBER_ATOMS)[np.newaxis, :, np.newaxis]
    axis = np.arange(3)[np.newaxis, np.newaxis, :]
    real = np.cos(np.pi * (mode + 1) * (3 * atom + axis) / NUMBER_MODES)
    imaginary = np.sin(np.pi * (mode + 1) * (3 * atom + axis + 0.5) / NUMBER_MODES)
    pattern = np.stack((real, imaginary), axis=-1)
    share = np.sum(pattern**2, axis=(-1, -2))
    scale = np.sqrt(mode_weights() / share)
    pattern = pattern * scale[:, :, np.newaxis, np.newaxis]
    return np.broadcast_to(pattern, (number_qpoints, *pattern.shape))


def _weight_per_axis():
    """The share of every mode that each atom carries along each axis."""
    return np.sum(_eigenvectors(1)[0] ** 2, axis=-1)


def dos_Sr2TiO4() -> raw.PhononDos:
    """Phonon density of states of Sr2TiO4, broadened from the same branches.

    The branches of :func:`band_Sr2TiO4` are evaluated on the q mesh and broadened, so
    the spectrum ends where the dispersion ends and the peaks sit at the flat optical
    branches. The projections use the displacement patterns of those same branches,
    which is what makes a projected density of states and a fat band tell one story.
    """
    energies, projections = _spectrum()
    return raw.PhononDos(
        energies=_demo.wrap_data(energies),
        dos=_demo.wrap_data(np.sum(projections, axis=(0, 1))),
        projections=_demo.wrap_data(projections),
        stoichiometry=_demo.stoichiometry.Sr2TiO4(has_ion_types=True),
    )


@functools.lru_cache(maxsize=1)
def _spectrum():
    """Energy axis and projections of the spectrum, broadened once.

    Every example in the documentation builds the demo data again in the same process,
    so broadening the mesh is cached. The arrays are read-only and
    :func:`py4vasp._demo.wrap_data` copies them, so no caller can reach the cache.
    """
    qpoints = kpoint.mesh()
    by_branch = branch_frequencies(qpoints)
    energies = np.linspace(
        0.0, np.max(by_branch) + ENERGY_MARGIN, showcase.NUMBER_POINTS
    )
    # dividing by the number of q points turns the sum over the mesh into an average, so
    # the spectrum integrates to the number of modes of a single cell
    per_mode = np.array(
        [
            showcase.broaden(
                energies, branch, weights=1 / len(qpoints), width=BROADENING
            )
            for branch in by_branch.T
        ]
    )
    projections = np.einsum("mad,me->ade", _weight_per_axis(), per_mode)
    energies.setflags(write=False)
    projections.setflags(write=False)
    return energies, projections


def mode_Sr2TiO4() -> raw.PhononMode:
    """Phonon modes of Sr2TiO4 at the zone centre.

    The frequencies are the ones the dispersion reports at Gamma, so the three acoustic
    branches come out at exactly zero: they translate the whole crystal, which costs no
    energy. None of them is imaginary, which is what marks a structure as stable.
    """
    at_gamma = branch_frequencies(np.zeros((1, 3)))[0] / EV_TO_THZ
    # VASP stores the eigenvalues of the dynamical matrix as complex numbers so that an
    # unstable mode can be reported as an imaginary frequency
    complex_frequencies = at_gamma.astype(np.complex128)
    as_pairs_of_reals = complex_frequencies.view(np.float64).reshape(-1, 2)
    return raw.PhononMode(
        structure=structure.Sr2TiO4(),
        frequencies=_demo.wrap_data(as_pairs_of_reals),
        eigenvectors=_demo.wrap_data(_displacements()),
    )


def _displacements():
    """Real displacement pattern of every zone-centre mode, shape ``(mode, mode)``.

    A mode at Gamma is real, so the pattern is the real part of the one the dispersion
    carries, rescaled so that every atom keeps the share :func:`mode_weights` gives it.
    The trailing axis runs over the three directions of every atom in turn, as VASP
    flattens them.
    """
    mode = np.arange(NUMBER_MODES)[:, np.newaxis, np.newaxis]
    atom = np.arange(NUMBER_ATOMS)[np.newaxis, :, np.newaxis]
    axis = np.arange(3)[np.newaxis, np.newaxis, :]
    pattern = np.cos(np.pi * (mode + 1) * (3 * atom + axis) / NUMBER_MODES)
    scale = np.sqrt(mode_weights() / np.sum(pattern**2, axis=-1))
    return (pattern * scale[:, :, np.newaxis]).reshape(NUMBER_MODES, NUMBER_MODES)
