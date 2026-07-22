# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Grid-based Bader partitioning of a charge density into atomic basins."""
import copy
import itertools

import numpy as np

from py4vasp import exception
from py4vasp._third_party import view
from py4vasp._util import select

# the 26 neighbor offsets on a 3d grid (all combinations of -1, 0, 1 except origin)
_OFFSETS = np.array(
    [offset for offset in itertools.product((-1, 0, 1), repeat=3) if any(offset)]
)


class BaderAnalysis(view.Mixin):
    """Bader charge analysis of a density defined on a grid.

    This class partitions a density into atomic basins following the grid-based
    steepest-ascent algorithm. The density is provided at construction time, so
    the (comparatively expensive) partition is computed once and reused by all
    methods. The structure supplies the lattice vectors, atomic positions, and
    atom labels the analysis needs.

    Parameters
    ----------
    structure
        A structure handler providing the geometry of the system via
        ``lattice_vectors``, ``positions``, ``volume``, ``to_dict``, and
        ``to_view``.
    density : np.ndarray
        The density sampled on a 3d grid with shape ``(nx, ny, nz)``, in the
        orientation returned by :py:meth:`Density.to_numpy`.
    snap_to_atoms : bool
        VASP pseudo densities need not be peaked at the nuclei, so the raw maxima
        may sit in bonds or interstitial regions. If True (default), every
        maximum is snapped to its nearest atom and the basins are labeled by atom
        index. If False, the basins are labeled by an arbitrary index enumerating
        the distinct maxima.
    """

    def __init__(self, structure, density, snap_to_atoms=True):
        self._density = np.asarray(density)
        _raise_error_if_density_not_3d(self._density)
        # Materialize everything derived from the structure now, while the raw data
        # is still available. The analysis is typically returned to the user after
        # the file context has closed, so it must not read lazily loaded data later.
        lattice_vectors = np.asarray(structure.lattice_vectors())
        positions = np.asarray(structure.positions())
        self._names = list(structure.to_dict()["names"])
        self._view = structure.to_view()
        maxima = _resolve_to_maxima(_ascent_pointers(self._density, lattice_vectors))
        if snap_to_atoms:
            labels = _label_by_nearest_atom(
                maxima, self._density.shape, lattice_vectors, positions
            )
        else:
            labels = _label_consecutively(maxima)
        self._basins = labels.reshape(self._density.shape)

    def __str__(self):
        charges = self.charges()
        width = max(len(name) for name in charges)
        lines = [
            f"    {name:<{width}}   {charge:.4f}" for name, charge in charges.items()
        ]
        return "Bader charges:\n" + "\n".join(lines)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def basins(self):
        """Return the basin partition of the density.

        Returns
        -------
        np.ndarray
            An integer array of the same shape as the density assigning every grid
            point to a basin. With ``snap_to_atoms`` the label is the index of the
            nearest atom, otherwise an arbitrary index enumerating the maxima.
        """
        return self._basins

    def charges(self, density=None):
        """Integrate a density within each basin.

        Parameters
        ----------
        density : np.ndarray or None
            The density to integrate on the same grid as the analysis. If None
            (default), the density used to construct the basins is integrated,
            which yields the Bader charges. Passing a different density integrates
            that quantity within the existing basins, which is useful to combine
            basins defined by one density with values taken from another.

        Returns
        -------
        dict
            Maps every atom label to the value integrated within its basin. The
            density follows the VASP convention where ``sum(density) / density.size``
            equals the total number of electrons, so summing over all basins
            reproduces that total.
        """
        density = self._density if density is None else np.asarray(density)
        _raise_error_if_density_not_3d(density)
        _raise_error_if_shape_differs(density, self._density)
        totals = np.bincount(
            self._basins.ravel(), weights=density.ravel(), minlength=len(self._names)
        )
        return dict(zip(self._names, totals / density.size))

    def to_view(self, threshold=0.0, supercell=None):
        """Visualize the Bader basins as labeled domains within the structure.

        Parameters
        ----------
        threshold : float
            Grid points where the density is below ``threshold`` times the maximum
            density are assigned to the background domain ``0`` and hidden. Because
            the cutoff is relative to the maximum, it is independent of the grid
            sampling and of the total number of electrons. The default of ``0``
            keeps every point.
        supercell : int | np.ndarray | None
            If present the structure is replicated the specified number of times
            along each direction.

        Returns
        -------
        View
            A visualization where every grid point carries the index of the atom
            its basin was snapped to (``1`` to number of atoms), or ``0`` if it is
            below the threshold. The domains are labeled by the atom names.
        """
        domains = self._basins + 1
        domains[self._density < threshold * self._density.max()] = 0
        viewer = copy.copy(self._view)
        if supercell is not None:
            viewer.supercell = supercell
        viewer.grid_domains = [
            view.GridDomain(
                quantity=domains[np.newaxis],
                label="basins",
                labels=self._names,
            )
        ]
        return viewer


def analysis_from_selection(structure, grid_for_selection, selection, snap_to_atoms=True):
    """Build a :class:`BaderAnalysis` from a single selected density.

    ``grid_for_selection`` is a callable mapping a selection string to a dictionary
    of labeled grid arrays. Injecting it keeps this helper free of any coupling to
    a specific quantity (composition instead of inheritance).
    """
    grids = grid_for_selection(selection)
    _raise_error_if_not_single_density(grids)
    (density,) = grids.values()
    return BaderAnalysis(structure, density, snap_to_atoms=snap_to_atoms)


def charges_from_selection(
    structure, grid_for_selection, selection, snap_to_atoms=True, analysis=None
):
    """Integrate the selected density within Bader basins.

    The basins are taken from ``analysis`` if given, otherwise from an inner
    ``basins=Y`` selection (partitioning the density selected by ``Y``), otherwise
    from the integrated density itself. Returns ``{atom: charge}`` for a single
    selection or ``{selection: {atom: charge}}`` for several.
    """
    results = {}
    for parsed in select.Tree.from_selection(selection).selections():
        basin_source = _basin_source(parsed)
        _raise_error_if_basins_and_analysis(basin_source, analysis)
        integrand_selection = _remaining_selection(parsed)
        for label, density in grid_for_selection(integrand_selection).items():
            basis = _basis_analysis(
                structure, grid_for_selection, basin_source, analysis, density,
                snap_to_atoms,
            )
            results[label] = basis.charges(density)
    if len(results) == 1:
        return next(iter(results.values()))
    return results


def _basin_source(parsed):
    for part in parsed:
        if isinstance(part, select.Assignment) and part.left_operand == "basins":
            return part.right_operand
    return None


def _remaining_selection(parsed):
    parts = [part for part in parsed if not _is_basins_assignment(part)]
    return select.selections_to_string([parts]) if parts else None


def _is_basins_assignment(part):
    return isinstance(part, select.Assignment) and part.left_operand == "basins"


def _basis_analysis(
    structure, grid_for_selection, basin_source, analysis, density, snap_to_atoms
):
    if analysis is not None:
        return analysis
    if basin_source is not None:
        (basin_density,) = grid_for_selection(basin_source).values()
        return BaderAnalysis(structure, basin_density, snap_to_atoms=snap_to_atoms)
    return BaderAnalysis(structure, density, snap_to_atoms=snap_to_atoms)


def _label_by_nearest_atom(maxima, shape, lattice_vectors, positions):
    "Assign the basin of each maximum to the nearest atom (minimum image)."
    unique_maxima, inverse = np.unique(maxima, return_inverse=True)
    grid_indices = np.stack(np.unravel_index(unique_maxima, shape), axis=1)
    fractional = grid_indices / np.array(shape)
    distance = fractional[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distance = (distance + 0.5) % 1.0 - 0.5
    cartesian = distance @ lattice_vectors
    nearest_atom = np.argmin(np.sum(cartesian**2, axis=-1), axis=1)
    return nearest_atom[inverse]


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


def _raise_error_if_density_not_3d(density):
    if density.ndim != 3:
        raise exception.IncorrectUsage(
            "The density must be sampled on a 3d grid, but an array with "
            f"{density.ndim} dimensions was provided."
        )


def _raise_error_if_shape_differs(density, reference):
    if density.shape != reference.shape:
        raise exception.IncorrectUsage(
            f"The density has shape {density.shape} which does not match the grid "
            f"{reference.shape} used to construct the basins."
        )


def _raise_error_if_basins_and_analysis(basin_source, analysis):
    if basin_source is not None and analysis is not None:
        raise exception.IncorrectUsage(
            "Specify the basin-defining density either via a 'basins=' selection "
            "or via the bader_analysis argument, but not both."
        )


def _raise_error_if_not_single_density(grids):
    if len(grids) != 1:
        raise exception.IncorrectUsage(
            "The Bader analysis requires exactly one density to define the basins, "
            f"but the selection produced {len(grids)}."
        )
