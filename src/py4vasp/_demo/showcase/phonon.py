# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw
from py4vasp._demo.showcase import kpoint, stoichiometry, structure

NUMBER_ATOMS = 7
NUMBER_MODES = 3 * NUMBER_ATOMS
SOUND_VELOCITY = 22.0  # THz per unit of the reduced q vector, the slope at Gamma
# Frequency in THz that every optical branch reaches at the zone centre. Sr2TiO4 has
# eighteen of them, spread over the range an oxide occupies; the heavier the atoms
# involved, the lower the branch.
_OPTICAL_AT_GAMMA = (
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
    frequencies = np.concatenate(
        (_acoustic(coordinates), _optical(coordinates)), axis=1
    )
    return raw.PhononBand(
        dispersion=raw.Dispersion(qpoints, _demo.wrap_data(frequencies)),
        stoichiometry=_demo.stoichiometry.Sr2TiO4(has_ion_types=True),
        eigenvectors=_demo.wrap_data(_eigenvectors(len(coordinates))),
        primitive_positions=_demo.wrap_data(structure.ideal_positions()),
    )


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
    phases = np.arange(1, len(_OPTICAL_AT_GAMMA) + 1)
    modulation = np.cos(2 * np.pi * np.sum(reduced, axis=1))
    shape = (len(coordinates), len(_OPTICAL_AT_GAMMA))
    dispersion = _OPTICAL_WIDTH * np.outer(modulation - 1, np.sqrt(phases) / 4)
    return np.broadcast_to(_OPTICAL_AT_GAMMA, shape) + dispersion


def _eigenvectors(number_qpoints):
    """Normalized displacement patterns, one per mode and q point."""
    # a deterministic set of orthonormal-looking patterns: mode n displaces the atoms
    # along a smooth wave whose wavelength grows with n, normalized per mode
    mode = np.arange(NUMBER_MODES)[:, np.newaxis, np.newaxis]
    atom = np.arange(NUMBER_ATOMS)[np.newaxis, :, np.newaxis]
    axis = np.arange(3)[np.newaxis, np.newaxis, :]
    real = np.cos(np.pi * (mode + 1) * (3 * atom + axis) / NUMBER_MODES)
    imaginary = np.sin(np.pi * (mode + 1) * (3 * atom + axis + 0.5) / NUMBER_MODES)
    norm = np.sqrt(np.sum(real**2 + imaginary**2, axis=(-1, -2)))
    pattern = np.stack((real, imaginary), axis=-1) / norm[:, None, None, None]
    return np.broadcast_to(pattern, (number_qpoints, *pattern.shape))
