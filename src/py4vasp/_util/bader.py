# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Grid-based Bader partitioning of a charge density into atomic basins."""
import itertools

import numpy as np

# the 26 neighbor offsets on a 3d grid (all combinations of -1, 0, 1 except origin)
_OFFSETS = np.array(
    [offset for offset in itertools.product((-1, 0, 1), repeat=3) if any(offset)]
)


def basins(charge, lattice_vectors, positions=None):
    """Partition a charge-density grid into Bader basins.

    Every grid point is assigned to a basin by following the steepest ascent of
    the density until a local maximum is reached; all points reaching the same
    maximum form one basin.

    Parameters
    ----------
    charge : np.ndarray
        The charge density sampled on a 3d grid with shape ``(nx, ny, nz)``, in
        the orientation returned by :py:meth:`Density.to_numpy`.
    lattice_vectors : np.ndarray
        The ``3x3`` lattice vectors (rows) spanning the unit cell. They define the
        Cartesian distances between neighboring grid points, so that the ascent
        respects the anisotropy of a non-orthogonal cell.
    positions : np.ndarray or None
        Optional fractional coordinates of the atoms with shape ``(n_atoms, 3)``.
        VASP pseudo densities need not be peaked at the nuclei, so the raw maxima
        may sit in bonds or interstitial regions. If positions are provided, every
        maximum is snapped to its nearest atom and the resulting basins are
        labeled by atom index. If omitted, the basins are labeled by an arbitrary
        index enumerating the distinct maxima.

    Returns
    -------
    np.ndarray
        An integer array of the same shape as ``charge`` assigning every grid
        point to a basin.
    """
    charge = np.asarray(charge)
    pointers = _ascent_pointers(charge, np.asarray(lattice_vectors))
    maxima = _resolve_to_maxima(pointers)
    labels = _label_consecutively(maxima)
    return labels.reshape(charge.shape)


def _resolve_to_maxima(pointers):
    "Follow the pointer field until each grid point reaches its maximum."
    while True:
        next_pointers = pointers[pointers]
        if np.array_equal(next_pointers, pointers):
            return pointers
        pointers = next_pointers


def _label_consecutively(maxima):
    "Map the flat maximum indices to consecutive basin labels starting at 0."
    _, labels = np.unique(maxima, return_inverse=True)
    return labels.reshape(maxima.shape)


def _ascent_pointers(charge, lattice_vectors):
    """For every grid point, return the flat index of its steepest-ascent neighbor.

    The gradient towards each of the 26 neighbors is the density difference divided
    by the Cartesian distance to that neighbor, so that the anisotropy of a
    non-orthogonal cell is accounted for. A grid point that is a local maximum
    points to itself.
    """
    shape = charge.shape
    spacing = lattice_vectors / np.reshape(shape, (3, 1))
    flat_indices = np.arange(charge.size).reshape(shape)
    best_slope = np.zeros(shape)
    pointers = flat_indices.copy()
    for offset in _OFFSETS:
        distance = np.linalg.norm(offset @ spacing)
        shift = tuple(-offset)
        slope = (np.roll(charge, shift, axis=(0, 1, 2)) - charge) / distance
        neighbor = np.roll(flat_indices, shift, axis=(0, 1, 2))
        improved = slope > best_slope
        best_slope = np.where(improved, slope, best_slope)
        pointers = np.where(improved, neighbor, pointers)
    return pointers.ravel()
