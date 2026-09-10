# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""A smooth field on the real-space grid the showcase quantities are sampled on.

VASP writes the density, the potential and the partial charge on a Fourier grid, and
py4vasp draws an isosurface of them. An isosurface of random numbers crosses noise at
almost every voxel and renders as confetti, so the showcase needs a field that is
smooth, periodic in the lattice, and peaked where the atoms are. The one here is a
promolecule: a superposition of atom-centred Gaussians, which is what a charge density
looks like before the atoms are allowed to bond.
"""
import functools
import itertools

import numpy as np

# Distance in Angstrom between neighbouring points of the grid. A calculation with a
# plane-wave cutoff around 300 eV gives about this, and it resolves the corrugation of a
# surface finely enough for the tip broadening of a simulated microscope to matter.
TARGET_SPACING = 0.10


def grid_for(lattice_vectors, spacing: float = TARGET_SPACING) -> tuple:
    """Number of grid points along each lattice vector, sampling the cell equally.

    Parameters
    ----------
    lattice_vectors
        The cell to fill, its vectors in the rows.
    spacing
        Distance the points should have. The result is the closest odd number of points
        to it, so the actual spacing differs a little.

    Returns
    -------
    -
        One count per lattice vector. They are odd because a Fourier transform of an
        even grid does not treat a reciprocal vector and its negative alike unless the
        cell is orthogonal, which leaves every operator derived from it asymmetric.
    """
    counts = interplanar_spacing(lattice_vectors) / spacing
    return tuple(2 * np.rint((counts - 1) / 2).astype(int) + 1)


def interplanar_spacing(lattice_vectors) -> np.ndarray:
    """Distance between neighbouring lattice planes along each lattice vector.

    This, and not the length of the lattice vector, is what a grid has to sample: the
    primitive cell of a body-centred lattice has three vectors of the same length that
    span very different distances between the planes they define.
    """
    volume = np.abs(np.linalg.det(lattice_vectors))
    areas = np.linalg.norm(
        np.cross(
            np.roll(lattice_vectors, -1, axis=0), np.roll(lattice_vectors, -2, axis=0)
        ),
        axis=1,
    )
    return volume / areas


def fractional_grid(grid) -> np.ndarray:
    """Fractional coordinates of every grid point, shape ``(*grid, 3)``."""
    axes = [np.arange(count) / count for count in grid]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def gaussian_sum(centers, weights, widths, lattice_vectors, grid) -> np.ndarray:
    """A normalized Gaussian around every centre, summed over the periodic images.

    Parameters
    ----------
    centers
        Fractional coordinates of the atoms, shape ``(atom, 3)``.
    weights
        How much each Gaussian contributes, e.g. the number of electrons of the atom.
    widths
        Standard deviation of each Gaussian in Angstrom.
    lattice_vectors
        The cell, its vectors in the rows.
    grid
        Number of points along each lattice vector, from :func:`grid_for`.

    Returns
    -------
    -
        The field on the grid, in units of the weights per cubic Angstrom. Because the
        Gaussians are normalized, its mean times the volume of the cell is the sum of
        the weights exactly.

        The images are summed rather than reduced to the closest one. That keeps the
        field exactly periodic *and* smooth: the distance to the closest image has a
        kink on the plane where two images are equally close, and such a kink shows up
        as a crease in a contour plot.
    """
    # normalization is only exact if every image that still contributes is included
    return _cached_sum(
        _hashable(centers),
        _hashable(weights),
        _hashable(widths),
        _hashable(lattice_vectors),
        tuple(grid),
    )


def _hashable(array):
    """Make an argument usable as a cache key without copying it back and forth."""
    array = np.ascontiguousarray(array, dtype=np.float64)
    return array.shape, array.tobytes()


def _from_hashable(hashed):
    shape, data = hashed
    return np.frombuffer(data, dtype=np.float64).reshape(shape)


@functools.lru_cache(maxsize=8)
def _cached_sum(centers, weights, widths, lattice_vectors, grid):
    """Evaluate the field once per set of arguments.

    Every example in the documentation builds the demo data again in the same process,
    and evaluating a grid this fine is the most expensive thing the showcase does. The
    result is read-only, and the producers hand it to
    :func:`py4vasp._demo.wrap_data`, which copies, so no caller can reach the cache.
    """
    centers = _from_hashable(centers)
    weights = _from_hashable(weights)
    widths = _from_hashable(widths)
    lattice_vectors = _from_hashable(lattice_vectors)
    points = fractional_grid(grid)
    field = np.zeros(grid)
    shifts = _shifts(lattice_vectors, np.max(widths)) @ lattice_vectors
    for center, weight, width in zip(centers, weights, widths):
        offset = points - center
        # reduce into the fractional unit box first, so that the images below cover the
        # whole neighbourhood of every point
        offset -= np.rint(offset)
        cartesian = offset @ lattice_vectors
        for shift in shifts:
            difference = cartesian + shift
            squared = np.einsum("...i,...i->...", difference, difference)
            field += weight * _normalized_gaussian(squared, width)
    field.setflags(write=False)
    return field


def _shifts(lattice_vectors, width):
    """Integer translations of every image a Gaussian of this width still reaches into.

    Six standard deviations leave a factor of a hundred million, so the normalization
    is exact to the precision that matters.

    How many images that is follows the spacing of the lattice planes, not the length of
    the lattice vectors, and the half accounts for the reduction into the unit box: no
    point is further from a centre than half the spacing, so a slab twenty Angstrom tall
    and two and a half wide needs several images in the plane and none across it.
    """
    spacing = interplanar_spacing(lattice_vectors)
    counts = np.maximum(np.ceil(6 * width / spacing - 0.5), 0).astype(int)
    ranges = [np.arange(-count, count + 1) for count in counts]
    return np.array(list(itertools.product(*ranges)))


def _normalized_gaussian(squared_distance, width):
    """The Gaussian, taking the square of the distance so that no root is needed."""
    normalization = (width * np.sqrt(2 * np.pi)) ** 3
    return np.exp(-0.5 * squared_distance / width**2) / normalization
