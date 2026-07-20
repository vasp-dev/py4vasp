# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Neighbor list of all atom pairs within a cutoff radius, derived from a
:class:`~py4vasp.calculation.structure`."""

import copy
import itertools

import numpy as np

from py4vasp import exception
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    merge_strings,
    quantity,
)
from py4vasp._calculation.structure import StructureHandler
from py4vasp._util import import_, select

# scipy is only required for the full (not core) installation, so import it
# lazily; the k-d tree is only touched when a neighbor list is actually computed.
spatial = import_.optional("scipy.spatial")

# NeighborList owns no raw data of its own; it derives cell, positions, and atom
# types from the structure. Dispatch therefore accesses the "structure" schema
# entry, exactly like optics derives from "dielectric_function".
_DATA_QUANTITY = "structure"

# Relative tolerance to absorb floating-point noise when the cutoff is an exact
# multiple of the perpendicular cell width, so ceil does not spuriously add a
# whole extra shell of replicas. Far larger than det/norm rounding (~1e-15) and
# far smaller than any physically meaningful fraction of a replica.
_REPLICA_TOL = 1e-9


def _replica_counts(lattice_vectors, cutoff):
    """Number of periodic replicas needed along each lattice direction.

    For a cutoff radius, an atom may have neighbors in any cell whose nearest
    plane lies within the cutoff. The robust (tilted-cell safe) criterion uses
    the *perpendicular* distance between the lattice planes stacked along each
    direction, ``d_i = |det(A)| / |a_j x a_k|``, rather than the lattice-vector
    lengths ``|a_i|``. The number of replicas is ``ceil(cutoff / d_i)``.

    Parameters
    ----------
    lattice_vectors : np.ndarray
        The (3, 3) matrix whose rows are the lattice vectors in Å.
    cutoff : float
        The neighbor cutoff radius in Å.

    Returns
    -------
    np.ndarray
        Three integers, the replica count along each lattice direction.
    """
    lattice_vectors = np.asarray(lattice_vectors)
    volume = np.abs(np.linalg.det(lattice_vectors))
    # cross products of the other two vectors: a1xa2, a2xa0, a0xa1
    cross = np.cross(lattice_vectors[[1, 2, 0]], lattice_vectors[[2, 0, 1]])
    perpendicular_width = volume / np.linalg.norm(cross, axis=1)
    return np.ceil(cutoff / perpendicular_width - _REPLICA_TOL).astype(int)


def _selection_label(selection):
    return " ".join(str(part) for part in selection)


def _part_mask(part, elements, source, neighbor):
    """Boolean mask selecting the pairs that match one selection element."""
    if isinstance(part, select.Group) and part.separator == select.pair_separator:
        source_type, neighbor_type = part.group
        _raise_if_unknown(source_type, elements)
        _raise_if_unknown(neighbor_type, elements)
        return (source == source_type) & (neighbor == neighbor_type)
    if isinstance(part, str):
        _raise_if_unknown(part, elements)
        return source == part
    message = (
        f"The selection '{part}' is not supported. Please select pairs of atom "
        "types with a tilde, e.g. 'Sr~Ti', or a single atom type, e.g. 'Sr'."
    )
    raise exception.IncorrectUsage(message)


def _raise_if_unknown(atom_type, elements):
    if not np.any(elements == atom_type):
        available = ", ".join(dict.fromkeys(elements))
        message = (
            f"The atom type '{atom_type}' is not present in the structure. "
            f"The available atom types are: {available}."
        )
        raise exception.IncorrectUsage(message)


class NeighborListHandler:
    """Computes the neighbor list from a single raw.Structure object."""

    def __init__(self, raw_structure, steps=None):
        self._structure = StructureHandler.from_data(raw_structure, steps=steps)

    @classmethod
    def from_data(cls, raw_structure, steps=None) -> "NeighborListHandler":
        return cls(raw_structure, steps=steps)

    def to_dict(self, selection=None, *, cutoff) -> dict:
        """Compute the neighbor list and store it in a dictionary.

        Without a selection the flat dictionary of all pairs is returned. When a
        selection is given, the result is keyed by the selection label and each
        value is the flat dictionary restricted to that pair of atom types.
        """
        pairs = self._all_pairs(cutoff)
        if selection is None:
            return pairs
        elements = np.array(self._structure._stoichiometry().elements())
        tree = select.Tree.from_selection(selection)
        return {
            _selection_label(sel): self._filter_pairs(pairs, sel, elements)
            for sel in tree.selections()
        }

    def __str__(self) -> str:
        elements = self._structure._stoichiometry().elements()
        atom_types = ", ".join(dict.fromkeys(elements))
        return f"neighbor list of {len(elements)} atoms ({atom_types})"

    def selections(self) -> list:
        """Return every pair of atom types that can be selected.

        Pair selection is directed (``'Si~C'`` keeps Si→C neighbors, ``'C~Si'``
        keeps C→Si), so both orderings are listed. Iterating over the result
        therefore partitions the complete neighbor list without omission.
        """
        atom_types = self._structure._stoichiometry().ion_types_list()
        pairs = itertools.product(atom_types, repeat=2)
        return [f"{a}{select.pair_separator}{b}" for a, b in pairs]

    def _filter_pairs(self, pairs, selection, elements):
        source = elements[pairs["indices"][:, 0]]
        neighbor = elements[pairs["indices"][:, 1]]
        masks = [_part_mask(part, elements, source, neighbor) for part in selection]
        mask = np.logical_and.reduce(masks)
        return {key: value[mask] for key, value in pairs.items()}

    def _lattice_vectors(self):
        return np.asarray(self._structure.lattice_vectors())

    def _all_pairs(self, cutoff) -> dict:
        """All atom pairs within *cutoff*, respecting periodic boundaries.

        The search wraps the atoms into the unit cell, replicates them into every
        periodic image that could hold a neighbor (see :func:`_replica_counts`),
        and uses a k-d tree to find the pairs within the cutoff in N log N time.
        """
        positions = np.asarray(self._structure.positions())
        if positions.ndim != 2:
            message = (
                "Computing a neighbor list for multiple steps is not implemented. "
                "Please select a single step, e.g. neighbor_list[0]."
            )
            raise exception.NotImplemented(message)
        lattice_vectors = self._lattice_vectors()
        home = (positions % 1.0) @ lattice_vectors
        offsets = self._cell_offsets(lattice_vectors, cutoff)
        images = (home[:, np.newaxis, :] + offsets @ lattice_vectors).reshape(-1, 3)
        number_offsets = len(offsets)
        home_tree = spatial.cKDTree(home)
        image_tree = spatial.cKDTree(images)
        distance_matrix = home_tree.sparse_distance_matrix(
            image_tree, cutoff, output_type="coo_matrix"
        )
        source = distance_matrix.row
        image = distance_matrix.col
        neighbor = image // number_offsets
        offset = offsets[image % number_offsets]
        # exclude only the atom paired with its own home image (same atom, zero
        # offset); an atom still neighbors its own replicas at nonzero offsets,
        # and two distinct atoms sharing a position remain a genuine pair.
        keep = ~((neighbor == source) & np.all(offset == 0, axis=1))
        source, neighbor, offset = source[keep], neighbor[keep], offset[keep]
        return {
            "indices": np.stack([source, neighbor], axis=1),
            "distances": distance_matrix.data[keep],
            "distance_vectors": images[image[keep]] - home[source],
            "cell_offsets": offset,
        }

    @staticmethod
    def _cell_offsets(lattice_vectors, cutoff):
        counts = _replica_counts(lattice_vectors, cutoff)
        ranges = [np.arange(-count, count + 1) for count in counts]
        return np.array(list(itertools.product(*ranges)))


@quantity("neighbor_list")
class NeighborList:
    """The neighbor list contains all atom pairs within a cutoff radius.

    Given a cutoff radius, this class determines every pair of atoms that is
    closer than the cutoff, taking the periodic boundary conditions into account.
    The relevant data (cell, atom types, positions) is taken from the structure.
    """

    def __init__(self, source, quantity_name: str = "neighbor_list", steps=None):
        self._source = source
        self._quantity_name = quantity_name
        self._steps = steps

    @classmethod
    def from_data(cls, raw_structure) -> "NeighborList":
        """Create a NeighborList dispatcher from raw structure data."""
        return cls(source=DataSource(raw_structure))

    def __getitem__(self, steps) -> "NeighborList":
        new = copy.copy(self)
        new._steps = steps
        return new

    def _handler_factory(self, raw_data):
        return NeighborListHandler.from_data(raw_data, steps=self._steps)

    def read(self, selection=None, *, cutoff) -> dict:
        """Compute the neighbor list and store it in a dictionary.

        Parameters
        ----------
        selection : str
            Select pairs of atom types with a tilde, e.g. 'Sr~Ti'. Combine
            multiple selections with commas or whitespace. When no selection is
            given, all pairs are returned.
        cutoff : float
            The neighbor cutoff radius in Å. Only pairs closer than this radius
            are returned.

        Returns
        -------
        dict
            Contains the atom ``indices`` (i, j) of each pair, their
            ``distances``, the cartesian ``distance_vectors`` from i to j, and
            the integer ``cell_offsets`` locating the periodic image of j. When a
            selection is given the result is keyed by the selection label.

        Examples
        --------
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        Compute all atom pairs within a radius of 3 Å

        >>> neighbors = calculation.neighbor_list.read(cutoff=3.0)
        >>> sorted(neighbors)
        ['cell_offsets', 'distance_vectors', 'distances', 'indices']

        Restrict the neighbor list to Sr-Ti pairs

        >>> selection = calculation.neighbor_list.read("Sr~Ti", cutoff=3.5)
        >>> list(selection)
        ['Sr~Ti']
        >>> selection["Sr~Ti"]["distances"]
        array([...])
        """
        return merge_default(
            self._source,
            _DATA_QUANTITY,
            selection,
            self._handler_factory,
            NeighborListHandler.to_dict,
            cutoff=cutoff,
        )

    def to_dict(self, selection=None, *, cutoff) -> dict:
        """Convenient alias for :py:meth:`read`. Please read the documentation there."""
        return self.read(selection, cutoff=cutoff)

    def selections(self) -> list:
        """Return every pair of atom types that can be selected.

        Each entry is a valid ``selection`` argument for :py:meth:`read`, so you
        can iterate over the result to obtain the neighbor list of every atom-type
        pair separately. Pair selection is directed, so both orderings of a
        cross-type pair are listed and iterating partitions the complete neighbor
        list without omission.

        Returns
        -------
        list
            All ordered atom-type pairs of the structure as ``'X~Y'`` strings.

        Examples
        --------
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        >>> calculation.neighbor_list.selections()
        ['Sr~Sr', 'Sr~Ti', 'Sr~O', 'Ti~Sr', 'Ti~Ti', 'Ti~O', 'O~Sr', 'O~Ti', 'O~O']
        """
        return merge_default(
            self._source,
            _DATA_QUANTITY,
            None,
            self._handler_factory,
            NeighborListHandler.selections,
        )

    def __str__(self, selection=None) -> str:
        return merge_strings(
            self._source,
            _DATA_QUANTITY,
            selection,
            self._handler_factory,
            NeighborListHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")
