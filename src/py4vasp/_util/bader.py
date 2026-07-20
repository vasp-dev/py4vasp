# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Grid-based Bader partitioning of a charge density into atomic basins."""
import itertools

import numpy as np

# the 26 neighbor offsets on a 3d grid (all combinations of -1, 0, 1 except origin)
_OFFSETS = np.array(
    [offset for offset in itertools.product((-1, 0, 1), repeat=3) if any(offset)]
)


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
