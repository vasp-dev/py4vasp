# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Project trajectories onto interatomic distances and discretize the resulting path,
e.g., to prepare the IRCCAR and ICONST files for a slow-growth simulation."""

import dataclasses

import numpy as np

from py4vasp import exception


class ReactionPathHandler:
    """Computes reaction paths from a single raw.Structure object."""

    @dataclasses.dataclass
    class ReactionPath:
        """A path through the space of interatomic distances.

        Every row of the coordinates is one point of the path, every column one pair
        of atoms, so the path can describe an IRC, an MD trajectory, or any other
        sequence of structures.

        Parameters
        ----------
        labels : Sequence[str]
            A label for every pair of atoms, e.g. ``"C~H"``.
        atom_pairs : array_like of shape (pairs, 2)
            The 1-based indices of the two atoms of every pair, in the order VASP
            numbers the atoms in the POSCAR and the ICONST file.
        coordinates : array_like of shape (steps, pairs)
            The distance in Å between the atoms of every pair at every step.
        lambda_ : float or None
            The λ parameter of the path-based collective variable. Only a discretized
            path defines it; it is None otherwise.

        Examples
        --------
        >>> from py4vasp._calculation.reaction_path import ReactionPathHandler
        >>> path = ReactionPathHandler.ReactionPath(
        ...     labels=["C~H", "H~N"],
        ...     atom_pairs=[[1, 2], [2, 3]],
        ...     coordinates=[[1.07, 2.52], [1.60, 1.07], [2.45, 0.99]],
        ... )
        >>> path.coordinates.shape
        (3, 2)
        """

        labels: tuple
        atom_pairs: np.ndarray
        coordinates: np.ndarray
        lambda_: float | None = None

        def __post_init__(self):
            self.labels = tuple(self.labels)
            self.atom_pairs = np.asarray(self.atom_pairs, dtype=np.int_)
            self.coordinates = np.asarray(self.coordinates, dtype=np.float64)
            number_pairs = len(self.labels)
            if self.atom_pairs.shape != (number_pairs, 2):
                message = f"The atom pairs must have the shape ({number_pairs}, 2), one pair of atom indices for each of the {number_pairs} labels, but they have the shape {self.atom_pairs.shape}."
                raise exception.IncorrectUsage(message)
            if self.coordinates.ndim != 2 or self.coordinates.shape[1] != number_pairs:
                message = f"The coordinates must have the shape (steps, {number_pairs}), one distance for each of the {number_pairs} labels at every step, but they have the shape {self.coordinates.shape}."
                raise exception.IncorrectUsage(message)

        def reversed(self):
            """Return the same path traversed in the opposite direction.

            Use this to join the two branches of an IRC calculation: both start at the
            transition state, so one of them has to be reversed so that the joined path
            runs from the reactant over the transition state to the product.

            Returns
            -------
            ReactionPath
                A new path with the order of the steps reversed.

            Examples
            --------
            >>> from py4vasp._calculation.reaction_path import ReactionPathHandler
            >>> path = ReactionPathHandler.ReactionPath(
            ...     ["C~H"], [[1, 2]], [[1.07], [1.60], [2.45]]
            ... )
            >>> path.reversed().coordinates[:, 0]
            array([2.45, 1.6 , 1.07])
            """
            return dataclasses.replace(self, coordinates=self.coordinates[::-1])

        def __add__(self, other):
            """Join two paths, appending the steps of the second one to the first.

            Both paths must describe the same pairs of atoms in the same order. The
            joined path is no longer discretized, so it does not keep λ.

            Examples
            --------
            >>> from py4vasp._calculation.reaction_path import ReactionPathHandler
            >>> to_reactant = ReactionPathHandler.ReactionPath(
            ...     ["C~H"], [[1, 2]], [[1.20], [1.07]]
            ... )
            >>> to_product = ReactionPathHandler.ReactionPath(
            ...     ["C~H"], [[1, 2]], [[1.20], [2.45]]
            ... )
            >>> path = to_reactant.reversed() + to_product
            >>> path.coordinates[:, 0]
            array([1.07, 1.2 , 1.2 , 2.45])
            """
            if not isinstance(other, type(self)):
                return NotImplemented
            if not np.array_equal(self.atom_pairs, other.atom_pairs):
                message = f"Only paths over the same pairs of atoms can be joined, but one path uses the pairs {self.atom_pairs.tolist()} and the other {other.atom_pairs.tolist()}. Please select the same pairs in the same order for both paths."
                raise exception.IncorrectUsage(message)
            coordinates = np.concatenate([self.coordinates, other.coordinates])
            return dataclasses.replace(self, coordinates=coordinates, lambda_=None)
