# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import functools
from fractions import Fraction

import numpy as np

from py4vasp import exception
from py4vasp._data import base
from py4vasp._util import convert, documentation

_kpoints_selection = """\
selection : str, optional
    You can select "kpoints_opt" or "kpoints_wan" here, to read from those meshes instead
    of the default one defined by the KPOINTS file.
"""


class Kpoint(base.Refinery):
    """The **k** points used in the Vasp calculation.

    This class provides utility functionality to extract information about the
    **k** points used by Vasp. As such it is mostly used as a helper class for
    other postprocessing classes to extract the required information, e.g., to
    generate a band structure.
    """

    @base.data_access
    def __str__(self):
        text = f"""k-points
{len(self._raw_data.coordinates)}
reciprocal"""
        for kpoint, weight in zip(self._raw_data.coordinates, self._raw_data.weights):
            text += "\n" + f"{kpoint[0]} {kpoint[1]} {kpoint[2]}  {weight}"
        return text

    @base.data_access
    @documentation.format(selection=_kpoints_selection)
    def to_dict(self):
        """Read the **k** points data into a dictionary.

        Parameters
        ----------
        {selection}

        Returns
        -------
        dict
            Contains the coordinates of the **k** points (in crystal units) as
            well as their weights used for integrations. Moreover, some data
            specified in the input file of Vasp are transferred such as the mode
            used to generate the **k** points, the line length (if line mode was
            used), and any labels set for specific points.
        """
        return {
            "mode": self.mode(),
            "line_length": self.line_length(),
            "number_kpoints": self.number_kpoints(),
            "coordinates": self._raw_data.coordinates[:],
            "weights": self._raw_data.weights[:],
            "labels": self.labels(),
        }

    @base.data_access
    @documentation.format(selection=_kpoints_selection)
    def line_length(self):
        """Get the number of points per line in the Brillouin zone.

        Parameters
        ----------
        {selection}

        Returns
        -------
        int
            The number of points used to sample a single line.
        """
        if self.mode() == "line":
            return self._raw_data.number
        return self.number_kpoints()

    @base.data_access
    @documentation.format(selection=_kpoints_selection)
    def number_lines(self):
        """Get the number of lines in the Brillouin zone.

        Parameters
        ----------
        {selection}

        Returns
        -------
        int
            The number of lines the band structure contains. For regular meshes this is set to 1."""
        return self.number_kpoints() // self.line_length()

    @base.data_access
    @documentation.format(selection=_kpoints_selection)
    def number_kpoints(self):
        """Get the number of points in the Brillouin zone.

        Parameters
        ----------
        {selection}

        Returns
        -------
        int
            The number of points used to sample the Brillouin zone.
        """
        return len(self._raw_data.coordinates)

    @base.data_access
    @documentation.format(selection=_kpoints_selection)
    def distances(self):
        """Convert the coordinates of the **k** points into a one dimensional array

        For every line in the Brillouin zone, the distance between each **k** point
        and the start of the line is calculated. Then the distances of different
        lines are concatenated into a single list. This routine is mostly useful
        to plot data along high-symmetry lines like band structures.

        Parameters
        ----------
        {selection}

        Returns
        -------
        np.ndarray
            A reduction of the **k** points onto a one-dimensional array based
            on the distance between the points.
        """
        cell = _last_step(self._raw_data.cell.lattice_vectors)
        cartesian_kpoints = np.linalg.solve(cell, self._raw_data.coordinates[:].T).T
        kpoint_lines = np.split(cartesian_kpoints, self.number_lines())
        kpoint_norms = [_line_distances(line) for line in kpoint_lines]
        concatenate_distances = lambda current, addition: (
            np.concatenate((current, addition + current[-1]))
        )
        return functools.reduce(concatenate_distances, kpoint_norms)

    @base.data_access
    @documentation.format(selection=_kpoints_selection)
    def mode(self):
        """Get the **k**-point generation mode specified in the Vasp input file

        Parameters
        ----------
        {selection}

        Returns
        -------
        str
            A string representing which mode was used to setup the k-points.
        """
        mode = convert.text_to_string(self._raw_data.mode).strip() or "# empty string"
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
                f"Could not understand the mode '{mode}' when refining the raw kpoints data."
            )

    @base.data_access
    @documentation.format(selection=_kpoints_selection)
    def labels(self):
        """Get any labels given in the input file for specific **k** points.

        Parameters
        ----------
        {selection}

        Returns
        -------
        list[str]
            A list of all the k-points explicitly named in the file or the coordinates of the
            band edges if no name was provided.
        """
        if not self._raw_data.label_indices.is_none():
            return self._labels_from_file()
        elif self.mode() == "line":
            return self._labels_at_band_edges()
        else:
            return None

    @base.data_access
    @documentation.format(selection=_kpoints_selection)
    def path_indices(self, start, finish):
        """Find linear dependent k points between start and finish

        Loop over all possible k points and return the indices of the ones for which
        k-point - start is linear dependent on finish - start.

        Parameters
        ----------
        {selection}

        Returns
        -------
        list[int]
            A list of indices of all k points that fulfil the linear dependence.
        """
        direction = np.array(finish) - np.array(start)
        deltas = self._raw_data.coordinates - np.array(start)
        areas = np.linalg.norm(np.cross(direction, deltas), axis=1)
        return np.flatnonzero(np.isclose(areas, 0))

    def _labels_from_file(self):
        labels = [""] * len(self._raw_data.coordinates)
        for label, index in zip(self._raw_data.labels, self._raw_indices()):
            labels[index] = convert.text_to_string(label.strip())
        return labels

    def _raw_indices(self):
        indices = np.array(self._raw_data.label_indices)
        if self.mode() == "line":
            line_length = self.line_length()
            return line_length * (indices // 2) - (indices + 1) % 2
        else:
            return indices - 1  # convert from Fortran to Python indices

    def _labels_at_band_edges(self):
        line_length = self.line_length()
        band_edge = lambda index: not (0 < index % line_length < line_length - 1)
        return [
            _kpoint_label(kpoint) if band_edge(index) else ""
            for index, kpoint in enumerate(self._raw_data.coordinates)
        ]


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
    fractions = [_to_latex(coordinate) for coordinate in kpoint]
    return f"$[{fractions[0]} {fractions[1]} {fractions[2]}]$"


def _to_latex(float):
    fraction = Fraction.from_float(float).limit_denominator()
    if fraction.denominator == 1:
        return str(fraction.numerator)
    else:
        return f"\\frac{{{fraction.numerator}}}{{{fraction.denominator}}}"
