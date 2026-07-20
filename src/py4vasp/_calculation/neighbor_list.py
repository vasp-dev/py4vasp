# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Neighbor list of all atom pairs within a cutoff radius, derived from a
:class:`~py4vasp.calculation.structure`."""

import numpy as np

from py4vasp._calculation.dispatch import DataSource, quantity
from py4vasp._calculation.structure import StructureHandler

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


class NeighborListHandler:
    """Computes the neighbor list from a single raw.Structure object."""

    def __init__(self, raw_structure, steps=None):
        self._structure = StructureHandler.from_data(raw_structure, steps=steps)

    @classmethod
    def from_data(cls, raw_structure, steps=None) -> "NeighborListHandler":
        return cls(raw_structure, steps=steps)

    def _lattice_vectors(self):
        return np.asarray(self._structure.lattice_vectors())


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

    def _handler_factory(self, raw_data):
        return NeighborListHandler.from_data(raw_data, steps=self._steps)
