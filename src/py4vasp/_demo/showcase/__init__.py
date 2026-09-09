# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Deterministic demo data shaped for presentation rather than for testing.

:mod:`py4vasp._demo` produces the data behind the ``raw_data`` test fixture: unseeded
random numbers and index ramps. Those discriminate well in an assertion but make a poor
figure, and a user who copies an example out of the documentation sees the figure. This
package produces the data behind :func:`py4vasp.demo.calculation` instead, so that every
example plots like a real VASP calculation.

Being a separate source of data, it is free to choose its own array shapes; the
constants below are deliberately larger than their counterparts in :mod:`py4vasp._demo`.
It reuses the producers of :mod:`py4vasp._demo` wherever those are already
deterministic, which is why :mod:`py4vasp._demo` deliberately does *not* import this
package -- the dependency runs one way only.
"""

import numpy as np

# constants for the shape of presentation data
NUMBER_POINTS = 301  # samples of an energy axis (density of states, optics)
LINE_LENGTH = 41  # k points along one segment of a band-structure path
NUMBER_STEPS = 12  # ionic steps of a relaxation trajectory
KPOINT_GRID = (8, 8, 8)  # k mesh a density of states is integrated over
# constants for the shape of a curve
BROADENING = 0.15  # standard deviation of a Gaussian in eV
CONVERGENCE_RATE = 0.45  # exponential decay per step of a relaxation


def broaden(energies, levels, weights=None, width=BROADENING):
    """Spread discrete levels into a smooth spectrum with normalized Gaussians.

    Parameters
    ----------
    energies
        Energy axis the spectrum is evaluated on.
    levels
        Discrete levels such as eigenvalues. Any shape is accepted and flattened, so
        the eigenvalues of all k points and bands can be passed in one call.
    weights
        Contribution of every level, broadcast against *levels*. Defaults to one state
        per level.
    width
        Standard deviation of the Gaussian in the unit of *energies*.

    Returns
    -------
    -
        Spectrum with the shape of *energies*. Because the Gaussians are normalized, it
        integrates to the total weight provided the levels are inside the energy axis.
    """
    levels = np.asarray(levels, dtype=np.float64)
    weights = np.ones(levels.shape) if weights is None else weights
    weights = np.broadcast_to(weights, levels.shape)
    distance = (np.atleast_1d(energies)[:, np.newaxis] - levels.ravel()) / width
    normalization = width * np.sqrt(2 * np.pi)
    return np.exp(-0.5 * distance**2) @ np.ravel(weights) / normalization


def converge(initial, final, number_steps=NUMBER_STEPS, rate=CONVERGENCE_RATE):
    """Approach a final value exponentially, the way a relaxation converges.

    Parameters
    ----------
    initial
        Value at the first step. May be an array, e.g. the positions of all atoms.
    final
        Value the trajectory converges onto. Broadcast against *initial*.
    number_steps
        Number of steps of the trajectory.
    rate
        Decay per step. The default leaves less than 1% of the initial deviation after
        :data:`NUMBER_STEPS` steps.

    Returns
    -------
    -
        Trajectory of shape ``(number_steps, *shape)`` starting exactly at *initial* and
        approaching *final* monotonically.
    """
    decay = np.exp(-rate * np.arange(number_steps))
    return final + np.multiply.outer(decay, np.subtract(initial, final))
