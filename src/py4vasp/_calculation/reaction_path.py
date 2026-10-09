# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Project trajectories onto interatomic distances and discretize the resulting path,
e.g., to prepare the IRCCAR and ICONST files for a slow-growth simulation."""

import dataclasses

import numpy as np

from py4vasp import exception

# Factor by which the targeted distance between successive points shrinks until every
# point of the discretized path lies at that distance within the tolerance.
_SHRINK_INCREMENT = 0.99


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

        def discretize(self, number_points, *, extra_points=0, tolerance):
            """Select points spaced approximately evenly along the path.

            A path-based collective variable, e.g., the IS coordinate of a slow-growth
            simulation, needs the path as a sequence of points that are about equally far
            apart in the space of the distances. This method picks such points among the
            steps of the path; it does not interpolate between them. It starts from the
            first step, targets the total length of the path divided by the number of
            intervals, and shrinks that target by 1% until every selected point lies at
            the targeted distance from its predecessor within the tolerance.

            Parameters
            ----------
            number_points : int
                How many points are selected along the path.
            extra_points : int
                How many points to add beyond each end of the path, continuing it
                linearly from the last two points. They keep the collective variable
                defined when the simulation runs slightly past the reactant or product.
                They count in addition to the selected points but not for λ.
            tolerance : float
                By how much, in Å, the distance between two successive points may
                deviate from the targeted one. A dense path, such as an IRC, permits a
                tight tolerance. If the tolerance is too tight for the steps of the path,
                the targeted distance shrinks a lot and the selected points stop short of
                the end of the path, so compare the last point to the last step. If the
                targeted distance shrinks below the tolerance, no evenly spaced points
                exist and an exception is raised.

            Returns
            -------
            ReactionPath
                The extra points before the path, the selected points, and the extra
                points after it. λ = 1 / ⟨d²⟩ is set from the distances d between
                successive selected points. Use λ for the IS line of the ICONST file; it makes
                the collective variable switch smoothly from one point to the next.

            Examples
            --------
            >>> import numpy as np
            >>> from py4vasp._calculation.reaction_path import ReactionPathHandler
            >>> x = np.linspace(0, 1, 101)
            >>> path = ReactionPathHandler.ReactionPath(["C~H"], [[1, 2]], x[:, np.newaxis])
            >>> discretized = path.discretize(6, tolerance=1e-3)
            >>> discretized.coordinates[:, 0]
            array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
            >>> round(discretized.lambda_, 6)
            25.0
            """
            _raise_if_invalid(
                len(self.coordinates), number_points, extra_points, tolerance
            )
            increment = _path_length(self.coordinates) / (number_points - 1)
            indices = None
            while indices is None:
                _raise_if_tolerance_missed(increment, tolerance)
                indices = _equidistant_indices(
                    self.coordinates, number_points, increment, tolerance
                )
                increment *= _SHRINK_INCREMENT
            points = self.coordinates[indices]
            return dataclasses.replace(
                self,
                coordinates=_extend(points, extra_points),
                lambda_=_suggest_lambda(points),
            )

        def to_IRCCAR(self):
            """Write the points of the path in the format of the IRCCAR file.

            The IRCCAR file defines the path for the path-based collective variable of
            a slow-growth or blue-moon simulation. Usually, you discretize the path
            first, see :meth:`discretize`, and use the same discretized path for the
            ICONST file, see :meth:`to_ICONST`. VASP expects the columns in the order of
            the R lines of the ICONST file.

            Returns
            -------
            str
                The number of points in the first line followed by one line per point
                with the distances in Å. Write it to a file named IRCCAR.

            Examples
            --------
            >>> from py4vasp._calculation.reaction_path import ReactionPathHandler
            >>> path = ReactionPathHandler.ReactionPath(
            ...     ["C~H", "H~N"], [[1, 2], [2, 3]], [[1.07, 2.52], [2.45, 0.99]]
            ... )
            >>> print(path.to_IRCCAR(), end="")
            2
             1.070000 2.520000
             2.450000 0.990000
            """
            lines = [str(len(self.coordinates))]
            lines += ["".join(f" {x:.6f}" for x in point) for point in self.coordinates]
            return "\n".join(lines) + "\n"


def _raise_if_invalid(number_steps, number_points, extra_points, tolerance):
    if not 2 <= number_points <= number_steps:
        message = f"The number of points must be at least 2 and at most the number of steps of the path ({number_steps}), but it is {number_points}."
        raise exception.IncorrectUsage(message)
    if extra_points < 0:
        message = f"The number of extra points must not be negative, but it is {extra_points}."
        raise exception.IncorrectUsage(message)
    if tolerance <= 0:
        message = f"The tolerance must be positive, but it is {tolerance}."
        raise exception.IncorrectUsage(message)


def _raise_if_tolerance_missed(increment, tolerance):
    if increment < tolerance:
        message = f"The steps of the path are too far apart to select points that are evenly spaced within the tolerance of {tolerance} Å. Please increase the tolerance, select fewer points, or provide a path with more steps."
        raise exception.IncorrectUsage(message)


def _path_length(coordinates):
    return np.sum(np.linalg.norm(np.diff(coordinates, axis=0), axis=1))


def _equidistant_indices(coordinates, number_points, increment, tolerance):
    """Walk along the path and pick the step closest to the targeted distance from the
    previously picked one. Return None if any of them misses it by the tolerance."""
    indices = [0]
    for _ in range(number_points - 1):
        start = indices[-1]
        distances = np.linalg.norm(coordinates[start:] - coordinates[start], axis=1)
        deviation = np.abs(distances - increment)
        closest = np.argmin(deviation)
        if deviation[closest] >= tolerance:
            return None
        indices.append(start + closest)
    return indices


def _extend(points, extra_points):
    before = (points[0] - points[1]) * np.arange(extra_points, 0, -1)[:, np.newaxis]
    after = (points[-1] - points[-2]) * np.arange(1, extra_points + 1)[:, np.newaxis]
    return np.concatenate([points[0] + before, points, points[-1] + after])


def _suggest_lambda(points):
    squared_spacing = np.sum(np.diff(points, axis=0) ** 2, axis=1)
    return float(1 / np.mean(squared_spacing))
