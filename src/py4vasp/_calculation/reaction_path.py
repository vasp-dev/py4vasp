# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Project trajectories onto interatomic distances and discretize the resulting path,
e.g., to prepare the IRCCAR and ICONST files for a slow-growth simulation."""

import collections.abc
import copy
import dataclasses
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
from py4vasp._third_party import graph
from py4vasp._util import select

# ReactionPath owns no raw data of its own; it derives the distances from the
# structure, so dispatch accesses the "structure" schema entry.
_DATA_QUANTITY = "structure"

# Offsets of the neighboring cells in which the closest image of an atom may lie.
_NEIGHBOR_CELLS = np.array(list(itertools.product((-1, 0, 1), repeat=3)))

# Factor by which the targeted distance between successive points shrinks until every
# point of the discretized path lies at that distance within the tolerance.
_SHRINK_INCREMENT = 0.99


class ReactionPathHandler:
    """Computes reaction paths from a single raw.Structure object."""

    @dataclasses.dataclass
    class Path(graph.Mixin, collections.abc.Sequence):
        """A path through the space of interatomic distances.

        Every row of the coordinates is one point of the path, every column one pair
        of atoms, so the path can describe an IRC, an MD trajectory, or any other
        sequence of structures. The path behaves like a sequence of its points:
        ``len(path)`` counts them, ``path[i]`` returns the distances of point i, and
        ``path[a:b]`` returns the part of the path between them as a new path. In
        particular, ``path[::-1]`` traverses the path in the opposite direction; use
        it to join the two branches of an IRC calculation, which both start at the
        transition state.

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
        >>> path = ReactionPathHandler.Path(
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

        def __add__(self, other):
            """Join two paths, appending the steps of the second one to the first.

            Both paths must describe the same pairs of atoms in the same order. The
            joined path is no longer discretized, so it does not keep λ.

            Examples
            --------
            >>> from py4vasp._calculation.reaction_path import ReactionPathHandler
            >>> to_reactant = ReactionPathHandler.Path(
            ...     ["C~H"], [[1, 2]], [[1.20], [1.07]]
            ... )
            >>> to_product = ReactionPathHandler.Path(
            ...     ["C~H"], [[1, 2]], [[1.20], [2.45]]
            ... )
            >>> path = to_reactant[::-1] + to_product
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

        def __len__(self):
            return len(self.coordinates)

        def __getitem__(self, index):
            if isinstance(index, slice):
                # a part of the path has a different spacing, so λ no longer fits
                coordinates = self.coordinates[index]
                return dataclasses.replace(self, coordinates=coordinates, lambda_=None)
            return self.coordinates[index]

        def __iter__(self):
            return iter(self.coordinates)

        def __contains__(self, point):
            return bool(np.any(self._matches(point)))

        def index(self, point, start=0, stop=None):
            """Return the index of the first point of the path equal to the given one.

            Parameters
            ----------
            point : array_like
                The distances in Å of every pair of atoms; they must match exactly.
            start, stop : int
                Search only the points between these indices, as for a list.

            Returns
            -------
            int
                The index of the first matching point.

            Raises
            ------
            ValueError
                If the point is not on the path.

            Examples
            --------
            >>> from py4vasp._calculation.reaction_path import ReactionPathHandler
            >>> path = ReactionPathHandler.Path(["C~H"], [[1, 2]], [[1.07], [1.60], [2.45]])
            >>> path.index([1.60])
            1
            """
            matches = np.flatnonzero(self._matches(point)[start:stop])
            if len(matches) == 0:
                raise ValueError(f"The point {point} is not on the path.")
            return int(matches[0]) + range(len(self))[start:stop].start

        def count(self, point):
            """Count how often the path passes through the given point.

            Parameters
            ----------
            point : array_like
                The distances in Å of every pair of atoms; they must match exactly.

            Returns
            -------
            int
                The number of points of the path equal to the given one.

            Examples
            --------
            >>> from py4vasp._calculation.reaction_path import ReactionPathHandler
            >>> path = ReactionPathHandler.Path(["C~H"], [[1, 2]], [[1.07], [1.60], [2.45]])
            >>> (path[::-1] + path).count([1.07])
            2
            """
            return int(np.sum(self._matches(point)))

        def _matches(self, point):
            point = np.asarray(point)
            if point.shape != self.coordinates.shape[1:]:
                return np.zeros(len(self), dtype=bool)
            return np.all(self.coordinates == point, axis=1)

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
            Path
                The extra points before the path, the selected points, and the extra
                points after it. λ = 1 / ⟨d²⟩ is set from the distances d between
                successive selected points. Use λ for the IS line of the ICONST file; it makes
                the collective variable switch smoothly from one point to the next.

            Examples
            --------
            >>> import numpy as np
            >>> from py4vasp._calculation.reaction_path import ReactionPathHandler
            >>> x = np.linspace(0, 1, 101)
            >>> path = ReactionPathHandler.Path(["C~H"], [[1, 2]], x[:, np.newaxis])
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
            >>> path = ReactionPathHandler.Path(
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

        def to_ICONST(self):
            """Write the ICONST file that defines the path-based collective variable.

            For every pair of atoms, the ICONST file gets an R line that defines the
            distance between them as a primitive coordinate. The final IS line combines
            them into the path-based collective variable with the λ of the discretized
            path, see :meth:`discretize`. Use it together with the IRCCAR file of the
            same discretized path, see :meth:`to_IRCCAR`; the λ only fits the points it
            was computed from. Every line ends with the status 0, which constrains the
            IS coordinate, as a slow-growth simulation with INCREM requires. Edit the
            status if your simulation needs a different one, see the ICONST page of the
            VASP wiki.

            Returns
            -------
            str
                The content of the ICONST file.

            Examples
            --------
            >>> import numpy as np
            >>> from py4vasp._calculation.reaction_path import ReactionPathHandler
            >>> x = np.linspace(0, 1, 101)
            >>> path = ReactionPathHandler.Path(
            ...     ["C~H", "H~N"], [[1, 2], [2, 3]], np.c_[x, 1 - x]
            ... )
            >>> print(path.discretize(6, tolerance=1e-3).to_ICONST(), end="")
            R 1 2 0
            R 2 3 0
            IS 12.5 12.5 0
            """
            if self.lambda_ is None:
                message = "The ICONST file needs the λ of a discretized path. Please call discretize first, and write the IRCCAR file from the same discretized path."
                raise exception.IncorrectUsage(message)
            lines = [f"R {first} {second} 0" for first, second in self.atom_pairs]
            lambdas = " ".join(len(self.atom_pairs) * [str(self.lambda_)])
            lines.append(f"IS {lambdas} 0")
            return "\n".join(lines) + "\n"

        def to_graph(self):
            """Plot the distance of every pair of atoms along the path.

            Use this to check how the bonds change from the reactant to the product,
            or which points a discretization selected.

            Returns
            -------
            Graph
                One line per pair of atoms with the distance in Å against the index of
                the point along the path.

            Examples
            --------
            >>> from py4vasp._calculation.reaction_path import ReactionPathHandler
            >>> path = ReactionPathHandler.Path(
            ...     ["C~H", "H~N"], [[1, 2], [2, 3]], [[1.07, 2.52], [2.45, 0.99]]
            ... )
            >>> path.to_graph()
            Graph(series=[Series(..., label='C~H', ...), Series(..., label='H~N', ...)], ...)
            """
            steps = np.arange(len(self.coordinates))
            series = [
                graph.Series(x=steps, y=distances, label=label)
                for label, distances in zip(self.labels, self.coordinates.T)
            ]
            return graph.Graph(series, xlabel="Step", ylabel="Distance (Å)")

    def __init__(self, raw_structure, steps=slice(None)):
        self._structure = StructureHandler.from_data(raw_structure, steps=steps)

    @classmethod
    def from_data(cls, raw_structure, steps=slice(None)) -> "ReactionPathHandler":
        return cls(raw_structure, steps=steps)

    def __str__(self) -> str:
        return f"""\
reaction path through {self._number_steps()} steps of {self._structure._stoichiometry()}
select pairs of atoms by their index, e.g. '1~2', or by element if it occurs once"""

    def to_dict(self, selection=None) -> dict:
        path = self.to_path(selection)
        return dict(zip(path.labels, path.coordinates.T))

    def to_path(self, selection=None) -> "ReactionPathHandler.Path":
        if not selection:
            message = "Please select the pairs of atoms whose distances define the path, e.g. '1~2, 1~3'. Use `selections` to list all pairs."
            raise exception.IncorrectUsage(message)
        selections = list(select.Tree.from_selection(selection).selections())
        labels = [_selection_label(selection) for selection in selections]
        atom_pairs = [self._atom_pair(selection) for selection in selections]
        coordinates = np.array([self._distances(*pair) for pair in atom_pairs]).T
        return self.Path(labels, atom_pairs, coordinates)

    def selections(self) -> list:
        number_atoms = self._structure.number_atoms()
        pairs = itertools.combinations(range(1, number_atoms + 1), 2)
        return [f"{first}{select.pair_separator}{second}" for first, second in pairs]

    def _atom_pair(self, selection):
        _raise_if_not_pair(selection)
        elements = self._structure._stoichiometry().elements()
        first, second = (_atom_index(atom, elements) for atom in selection[0].group)
        if first == second:
            message = f"The selection '{_selection_label(selection)}' measures the distance of atom {first} to itself. Please select two different atoms."
            raise exception.IncorrectUsage(message)
        return [first, second]

    def _distances(self, first, second):
        positions = _all_steps(np.asarray(self._structure.positions()), ndim=3)
        lattice_vectors = _all_steps(self._structure.lattice_vectors(), ndim=3)
        difference = positions[:, second - 1] - positions[:, first - 1]
        difference -= np.rint(difference)
        images = difference[:, np.newaxis, :] + _NEIGHBOR_CELLS
        cartesian = images @ lattice_vectors
        return np.min(np.linalg.norm(cartesian, axis=-1), axis=1)

    def _number_steps(self):
        return len(_all_steps(np.asarray(self._structure.positions()), ndim=3))


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


def _selection_label(selection):
    return " ".join(str(part) for part in selection)


def _raise_if_not_pair(selection):
    is_pair = (
        len(selection) == 1
        and isinstance(selection[0], select.Group)
        and selection[0].separator == select.pair_separator
        and len(selection[0].group) == 2
        and all(isinstance(atom, str) for atom in selection[0].group)
    )
    if not is_pair:
        message = f"The selection '{_selection_label(selection)}' is not a pair of atoms. Please join exactly two atoms with a tilde, e.g. '1~2' or 'C~H', and separate pairs with commas."
        raise exception.IncorrectUsage(message)


def _atom_index(atom, elements):
    if atom.isdecimal():
        return _check_index(int(atom), len(elements))
    indices = [index + 1 for index, element in enumerate(elements) if element == atom]
    if len(indices) == 1:
        return indices[0]
    if not indices:
        available = ", ".join(dict.fromkeys(elements))
        message = f"The element '{atom}' is not present in the structure. The available elements are: {available}."
    else:
        message = f"The element '{atom}' occurs {len(indices)} times in the structure, at the atoms {indices}. Please select the atom by its index instead."
    raise exception.IncorrectUsage(message)


def _check_index(index, number_atoms):
    if not 1 <= index <= number_atoms:
        message = f"The atom index {index} is out of range. Atoms are counted from 1 in the order of the POSCAR file, and the structure has {number_atoms} atoms."
        raise exception.IncorrectUsage(message)
    return index


def _all_steps(array, ndim):
    # a single selected step lacks the leading axis of the steps
    return array if array.ndim == ndim else array[np.newaxis]


@quantity("reaction_path")
class ReactionPath:
    """The reaction path follows the distances between pairs of atoms through a run.

    Use it to prepare a slow-growth or blue-moon simulation along the intrinsic
    reaction coordinate (IRC). Every VASP run with IBRION = 40 follows the IRC from the
    transition state to one of the minima. Map each of the two runs onto the
    distances that define the reaction, join them into one path from the reactant
    over the transition state to the product, select evenly spaced points, and write
    the IRCCAR and ICONST files. The data is taken from the structures of the run, so
    the same works for the trajectory of a molecular-dynamics run.

    Atoms are counted from 1 in the order of the POSCAR file, as in the ICONST file.
    Distances are in Å and take the closest periodic image of the second atom.

    Examples
    --------
    Prepare the IRCCAR and ICONST files from the two branches of an IRC calculation in
    the directories irc/m and irc/p, here for the HCN → HNC isomerization with the
    atoms C, H, and N::

        import py4vasp
        to_reactant = py4vasp.Calculation.from_path("irc/m").reaction_path
        to_product = py4vasp.Calculation.from_path("irc/p").reaction_path
        pairs = "C~H, C~N, H~N"
        path = to_reactant.to_path(pairs)[::-1] + to_product.to_path(pairs)
        discretized = path.discretize(15, extra_points=2, tolerance=5e-3)
        with open("IRCCAR", "w") as file:
            file.write(discretized.to_IRCCAR())
        with open("ICONST", "w") as file:
            file.write(discretized.to_ICONST())

    Both runs start at the transition state, so the run toward the reactant is
    reversed before the run toward the product is appended.

    See Also
    --------
    py4vasp._calculation.structure.Structure :
        The positions and the cell the distances are derived from.
    py4vasp._calculation.neighbor_list.NeighborList :
        All pairs of atoms within a cutoff for a single step.
    """

    Path = ReactionPathHandler.Path

    # is_available checks the structure, which is where the data actually lives.
    _availability_quantity = _DATA_QUANTITY

    def __init__(self, source, quantity_name: str = "reaction_path", steps=slice(None)):
        self._source = source
        self._quantity_name = quantity_name
        self._steps = steps

    @classmethod
    def from_data(cls, raw_structure) -> "ReactionPath":
        """Create a ReactionPath from raw structure data, e.g., to test it."""
        return cls(source=DataSource(raw_structure))

    def __getitem__(self, steps) -> "ReactionPath":
        new = copy.copy(self)
        new._steps = steps
        return new

    def _handler_factory(self, raw_data):
        return ReactionPathHandler.from_data(raw_data, steps=self._steps)

    def read(self, selection=None) -> dict:
        """Read the distances between the selected pairs of atoms at every step.

        Parameters
        ----------
        selection : str
            The pairs of atoms joined by a tilde, e.g. '1~2', separated by commas. Give
            an atom by its index counted from 1 in the order of the POSCAR file, or by
            its element if the structure contains only one atom of that element.

        Returns
        -------
        dict
            For every selected pair, the distance in Å at every step. Index the
            quantity to restrict the steps, e.g. ``reaction_path[10:20]``; all steps
            are used by default.

        Examples
        --------
        First, we create some example data so that you can follow along. Alternatively,
        use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation()

        Read the distance between the Ti atom (index 3) and the first O atom (index 4)

        >>> calculation.reaction_path.read("3~4")
        {'3~4': array([...])}

        Atoms that occur only once can be selected by their element instead

        >>> calculation.reaction_path[0:2].read("Ti~4")
        {'Ti~4': array([..., ...])}
        """
        return merge_default(
            self._source,
            _DATA_QUANTITY,
            selection,
            self._handler_factory,
            ReactionPathHandler.to_dict,
        )

    def to_dict(self, selection=None) -> dict:
        """Convenient alias for :py:meth:`read`. Please read the documentation there."""
        return self.read(selection)

    def to_path(self, selection=None) -> "ReactionPathHandler.Path":
        """Map the run onto the distances between the selected pairs of atoms.

        Parameters
        ----------
        selection : str
            The pairs of atoms, see :py:meth:`read`. Their order sets the order of the
            columns of the IRCCAR file and of the R lines of the ICONST file.

        Returns
        -------
        Path
            The path through the space of the selected distances with one point per
            step. Join paths with ``+``, reverse them with ``[::-1]``, and select
            evenly spaced points with ``discretize()``.

        Examples
        --------
        First, we create some example data so that you can follow along. Alternatively,
        use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation()

        Map the run onto two distances; the path has one point per step

        >>> path = calculation.reaction_path.to_path("3~4, 1~2")
        >>> path.labels
        ('3~4', '1~2')
        >>> path.coordinates.shape
        (..., 2)
        """
        return merge_default(
            self._source,
            _DATA_QUANTITY,
            selection,
            self._handler_factory,
            ReactionPathHandler.to_path,
        )

    def selections(self) -> list:
        """Return every pair of atoms that can be selected.

        Each entry is a valid ``selection`` argument for :py:meth:`read` and
        :py:meth:`to_path`. Atoms that occur only once can also be selected by their
        element, e.g. 'C~H'.

        Returns
        -------
        list
            All pairs of atoms as '1~2' strings with atoms counted from 1.

        Examples
        --------
        First, we create some example data so that you can follow along. Alternatively,
        use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation()

        >>> calculation.reaction_path.selections()
        ['1~2', '1~3', '1~4', ...]
        """
        return merge_default(
            self._source,
            _DATA_QUANTITY,
            None,
            self._handler_factory,
            ReactionPathHandler.selections,
        )

    def print(self, selection: str | None = None) -> None:
        """Print a string representation of this quantity.

        Parameters
        ----------
        selection : str | None
            Select which source of the quantity is printed. If you select multiple
            sources, py4vasp prints one block per source.
        """
        print(self.__str__(selection))

    def __str__(self, selection=None) -> str:
        return merge_strings(
            self._source,
            _DATA_QUANTITY,
            selection,
            self._handler_factory,
            ReactionPathHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")
