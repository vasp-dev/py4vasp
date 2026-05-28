# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import functools
from contextlib import suppress
from fractions import Fraction
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from py4vasp import exception
from py4vasp._calculation.dispatch import DataSource, merge_default, merge_strings, quantity
from py4vasp._raw import data as raw
from py4vasp._raw.data_db import Kpoint_DB
from py4vasp._util import check, convert

_TO_DATABASE_SUPPRESSED_EXCEPTIONS = (
    exception.Py4VaspError,
    exception.RefinementError,
)


def _safe_call(func):
    with suppress(*_TO_DATABASE_SUPPRESSED_EXCEPTIONS):
        return func()
    return None


class KpointHandler:
    """Handler for k-point data."""

    def __init__(self, raw_kpoint: raw.Kpoint):
        self._raw_kpoint = raw_kpoint

    @classmethod
    def from_data(cls, raw_kpoint: raw.Kpoint) -> "KpointHandler":
        return cls(raw_kpoint)

    def __str__(self):
        text = f"""k-points
{len(self._raw_kpoint.coordinates)}
reciprocal"""
        for kpoint, weight in zip(self._raw_kpoint.coordinates, self._raw_kpoint.weights):
            text += "\n" + f"{kpoint[0]} {kpoint[1]} {kpoint[2]}  {weight}"
        return text

    def to_dict(self) -> dict[str, Any]:
        labels = self.labels()
        labels_dict = {} if labels is None else {"labels": labels}
        return {
            "mode": self.mode(),
            "line_length": self.line_length(),
            "number_kpoints": self.number_kpoints(),
            "coordinates": self._raw_kpoint.coordinates[:],
            "weights": self._raw_kpoint.weights[:],
            **labels_dict,
        }

    def to_database(self) -> dict:
        number_x = self._raw_kpoint.number_x
        number_y = self._raw_kpoint.number_y
        number_z = self._raw_kpoint.number_z
        has_grid = not (any(check.is_none(n) for n in (number_x, number_y, number_z)))
        grid_kpoints = (
            None
            if not has_grid
            else [number_x, number_y, number_z]
        )
        user_labels = None
        if not check.is_none(self._raw_kpoint.label_indices):
            user_labels = [k for k in self._labels_from_file() if k != ""]
            user_labels = None if len(user_labels) == 0 else user_labels
        sampled_points = sorted(set(user_labels)) if user_labels is not None else None
        mode = _safe_call(self.mode)
        line_length = _safe_call(self.line_length)
        num_kpoints_total = _safe_call(self.number_kpoints)
        num_lines = _safe_call(self.number_lines)
        return {
            "kpoint": Kpoint_DB(
                mode=mode,
                line_length=line_length,
                num_kpoints_total=num_kpoints_total,
                num_lines=num_lines,
                num_kpoints_grid=grid_kpoints,
                labels=user_labels,
                labels_unique=sampled_points,
            )
        }

    def line_length(self) -> int:
        if self.mode() == "line":
            return self._raw_kpoint.number
        return self.number_kpoints()

    def number_lines(self) -> int:
        return int(self.number_kpoints() // self.line_length())

    def number_kpoints(self) -> int:
        return len(self._raw_kpoint.coordinates)

    def distances(self) -> np.ndarray:
        cell = _last_step(self._raw_kpoint.cell.lattice_vectors)
        cartesian_kpoints = np.linalg.solve(cell, self._raw_kpoint.coordinates[:].T).T
        kpoint_lines = np.split(cartesian_kpoints, self.number_lines())
        kpoint_norms = [_line_distances(line) for line in kpoint_lines]
        concatenate_distances = lambda current, addition: (
            np.concatenate((current, addition + current[-1]))
        )
        return functools.reduce(concatenate_distances, kpoint_norms)

    def mode(self) -> str:
        mode = convert.text_to_string(self._raw_kpoint.mode).strip() or "# empty string"
        first_char = mode[0].lower()
        if first_char == "a":
            return "automatic"
        elif first_char == "b":
            return "generating lattice"
        elif first_char == "e":
            return "explicit"
        elif first_char == "g":
            return "gamma"
        elif first_char == "l":
            return "line"
        elif first_char == "m":
            return "monkhorst"
        else:
            raise exception.RefinementError(
                f"Could not understand the mode \'{mode}\' when refining the raw kpoints data."
            )

    def labels(self) -> list[str] | None:
        if not self._raw_kpoint.label_indices.is_none():
            return self._labels_from_file()
        elif self.mode() == "line":
            return self._labels_at_band_edges()
        else:
            return None

    def path_indices(self, start: ArrayLike, finish: ArrayLike) -> np.ndarray:
        direction = np.array(finish) - np.array(start)
        deltas = self._raw_kpoint.coordinates - np.array(start)
        areas = np.linalg.norm(np.cross(direction, deltas), axis=1)
        return np.flatnonzero(np.isclose(areas, 0))

    def _labels_from_file(self):
        labels = [""] * len(self._raw_kpoint.coordinates)
        for label, index in zip(self._raw_kpoint.labels, self._raw_indices()):
            labels[index] = convert.text_to_string(label.strip())
        return labels

    def _raw_indices(self):
        indices = np.array(self._raw_kpoint.label_indices)
        if self.mode() == "line":
            line_length = self.line_length()
            return line_length * (indices // 2) - (indices + 1) % 2
        else:
            return indices - 1

    def _labels_at_band_edges(self):
        line_length = self.line_length()
        band_edge = lambda index: not (0 < index % line_length < line_length - 1)
        return [
            _kpoint_label(kpoint) if band_edge(index) else ""
            for index, kpoint in enumerate(self._raw_kpoint.coordinates)
        ]

    def _reciprocal_lattice_vectors(self):
        scale = self._raw_kpoint.cell.scale
        lattice_vectors = scale * _last_step(self._raw_kpoint.cell.lattice_vectors)
        volume = np.linalg.det(lattice_vectors)
        return (2.0 * np.pi / volume) * np.array(
            [
                np.cross(lattice_vectors[1], lattice_vectors[2]),
                np.cross(lattice_vectors[2], lattice_vectors[0]),
                np.cross(lattice_vectors[0], lattice_vectors[1]),
            ]
        )


@quantity("kpoint")
class Kpoint:
    """The **k**-point mesh used in the VASP calculation.

    In VASP calculations, **k** points play an important role in discretizing the
    Brillouin zone of a crystal. For self-consistent DFT calculations, typically a
    regular grid of **k** points is employed to sample the Brillouin zone. A
    sufficiently dense **k**-points mesh is critical for the precision of your DFT
    calculation, so make sure to test the results for different meshes. Denser
    **k** point meshes provide more accurate results but also demand greater
    computational resources.

    Another common use case is irregular meshes in non-self-consistent calculations.
    In particular in band structure analysis, one employs a mesh along specific lines
    in the Brillouin zone. The line mode involves connecting high-symmetry points and
    calculating the electronic band structure along these paths.

    This class provides utility functionality to extract information about either of
    the aforementioned use cases. As such it is mostly used as a helper class for
    other postprocessing classes to extract the required information, e.g., to
    generate a band structure. It may also be used to programmatically analyze the
    selected **k** point mesh or take subsets along high symmetry lines.
    """

    def __init__(self, source, quantity_name="kpoint"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_kpoint):
        return cls(source=DataSource(raw_kpoint))

    def _handler_factory(self, raw):
        return KpointHandler.from_data(raw)

    def __str__(self):
        return merge_strings(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self, selection=None) -> dict:
        """Read the **k** points data into a dictionary.

        Parameters
        ----------
        selection : str, optional
            You can select "kpoints_opt" or "kpoints_wan" here, to read from those
            meshes instead of the default one defined by the KPOINTS file.

        Returns
        -------
        -
            Contains the coordinates of the **k** points (in crystal units) as
            well as their weights used for integrations. Moreover, some data
            specified in the input file of Vasp are transferred such as the mode
            used to generate the **k** points, the line length (if line mode was
            used), and any labels set for specific points.

        Examples
        --------
        Read the **k** points data into a dictionary:

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> calculation.kpoint.read()
        {'mode': ..., 'line_length': ..., 'number_kpoints': ..., 'coordinates': array(...), 'weights': array(...)}

        Select the **k** points from the "kpoints_opt" mesh instead of the default one:

        >>> calculation.kpoint.read(selection="kpoints_opt")
        {'mode': ..., 'line_length': ..., 'number_kpoints': ..., 'coordinates': array(...), 'weights': array(...)}
        """
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler.to_dict,
        )

    def to_dict(self, selection=None) -> dict:
        """Convenient alias for :py:meth:`read`. Please read the documentation there."""
        return self.read(selection=selection)

    def line_length(self) -> int:
        """Get the number of points per line in the Brillouin zone.

        Returns
        -------
        -
            The number of points used to sample a single line.

        Examples
        --------
        Get the number of points per line in the Brillouin zone:

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> calculation.kpoint.line_length()
        48
        """
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler.line_length,
        )

    def number_lines(self) -> int:
        """Get the number of lines in the Brillouin zone.

        Returns
        -------
        -
            The number of lines the band structure contains. For regular meshes this is
            set to 1.

        Examples
        --------
        Get the number of lines in the Brillouin zone:

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> calculation.kpoint.number_lines()
        4
        """
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler.number_lines,
        )

    def number_kpoints(self) -> int:
        """Get the number of points in the Brillouin zone.

        Returns
        -------
        -
            The number of points used to sample the Brillouin zone.

        Examples
        --------
        Get the number of points in the Brillouin zone:

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> calculation.kpoint.number_kpoints()
        48
        """
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler.number_kpoints,
        )

    def distances(self) -> np.ndarray:
        """Convert the coordinates of the **k** points into a one dimensional array.

        For every line in the Brillouin zone, the distance between each **k** point
        and the start of the line is calculated. Then the distances of different
        lines are concatenated into a single list. This routine is mostly useful
        to plot data along high-symmetry lines like band structures.

        Returns
        -------
        -
            A reduction of the **k** points onto a one-dimensional array based
            on the distance between the points.

        Examples
        --------
        Convert the coordinates of the **k** points into a one dimensional array:

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> calculation.kpoint.distances()
        array([...])
        """
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler.distances,
        )

    def mode(self) -> str:
        """Get the **k**-point generation mode specified in the Vasp input file.

        Returns
        -------
        -
            A string representing which mode was used to setup the k-points.

        Examples
        --------
        Get the **k**-point generation mode:

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> calculation.kpoint.mode()
        'line'
        """
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler.mode,
        )

    def labels(self) -> list[str] | None:
        """Get any labels given in the input file for specific **k** points.

        The returned labels depend on the **k**-point mode and whether the user
        provided explicit labels in the input file:

        - If labels are specified in the KPOINTS file, a list of strings is returned
          with one entry per **k** point. Points with no user-defined label get an
          empty string, while labeled points carry the name from the input file.
        - If line mode is used but no labels were given, VASP automatically assigns
          labels at the band edges. Interior points along the line receive an empty
          string. Labels are formatted as LaTeX fractions.
        - For any other mode without explicit labels (e.g., a regular Gamma or
          Monkhorst-Pack grid), ``None`` is returned.

        Returns
        -------
        -
            A list of strings (one per **k** point) or ``None`` if no labeling is
            applicable.

        Examples
        --------
        If no labels were given and line mode is not used, returns None:

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> result = calculation.kpoint.labels()
        >>> assert result is None

        If line mode is used, VASP automatically assigns labels to the band edges:

        >>> calculation.kpoint.labels()
        ['$[0 0 0]$', ...]
        """
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler.labels,
        )

    def path_indices(self, start: ArrayLike, finish: ArrayLike) -> np.ndarray:
        """Find linear dependent k points between start and finish.

        Loop over all possible k points and return the indices of the ones for which
        k-point - start is linear dependent on finish - start.

        The primary use case is to extract a band-structure-like slice from a
        regular **k**-point grid. In certain calculation types — such as
        time-dependent DFT (TDDFT), GW, or BSE — VASP only supports uniform
        Gamma or Monkhorst-Pack meshes. You can use this method to select all grid
        points that happen to lie on a high-symmetry path.

        Parameters
        ----------
        start
            The starting **k** point of the path segment in fractional (crystal)
            coordinates. Expects exactly 3 coordinates.
        finish
            The ending **k** point of the path segment in fractional (crystal)
            coordinates. Expects exactly 3 coordinates.

        Returns
        -------
        -
            An integer array of indices (into the full **k**-point list) of all
            **k** points that lie on the line segment from ``start`` to ``finish``.

        Examples
        --------
        Extract all **k** points on a line through the Brillouin zone:

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> start = [0, 0, 0.125]
        >>> finish = [1, 0, 0.125]
        >>> calculation.kpoint.path_indices(start, finish)
        array([...])
        """
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler.path_indices,
            start, finish,
        )

    def selections(self):
        from py4vasp._raw import definition as raw_module
        return {self._quantity_name: list(raw_module.selections(self._quantity_name))}

    def _reciprocal_lattice_vectors(self):
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, KpointHandler._reciprocal_lattice_vectors,
        )


def _last_step(lattice_vectors):
    if lattice_vectors.ndim == 2:
        return lattice_vectors
    else:
        return lattice_vectors[-1]


def _line_distances(coordinates):
    distances = np.zeros(len(coordinates))
    norms = np.linalg.norm(coordinates[1:] - coordinates[:-1], axis=1)
    distances[1:] = np.cumsum(norms)
    return distances


def _kpoint_label(kpoint):
    fractions = [convert.Fraction(coordinate).latex() for coordinate in kpoint]
    return f"$[{fractions[0]} {fractions[1]} {fractions[2]}]$"
