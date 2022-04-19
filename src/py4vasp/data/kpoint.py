# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp.data._base import DataBase, RefinementDescriptor
import py4vasp._util.documentation as _documentation
import py4vasp.exceptions as exception
import py4vasp._util.convert as _convert
import functools
from fractions import Fraction
import numpy as np


_kpoints_opt_source = """
source : str, optional
    If you used a KPOINTS_OPT file to use a second k-point mesh, you can provide
    a keyword argument `source="kpoints_opt"` to use the k-points defined in that
    file instead of the one specified in KPOINTS.
""".strip()


class Kpoint(DataBase):
    """The **k** points used in the Vasp calculation.

    This class provides utility functionality to extract information about the
    **k** points used by Vasp. As such it is mostly used as a helper class for
    other postprocessing classes to extract the required information, e.g., to
    generate a band structure.

    Parameters
    ----------
    raw_kpoint : RawKpoint
        Dataclass containing the raw **k**-points data used in the calculation.
    """

    to_dict = RefinementDescriptor("_to_dict")
    read = RefinementDescriptor("_to_dict")
    line_length = RefinementDescriptor("_line_length")
    number_lines = RefinementDescriptor("_number_lines")
    distances = RefinementDescriptor("_distances")
    mode = RefinementDescriptor("_mode")
    labels = RefinementDescriptor("_labels")
    __str__ = RefinementDescriptor("_to_string")

    def _to_string(self):
        text = f"""k-points
{len(self._raw_data.coordinates)}
reciprocal"""
        for kpoint, weight in zip(self._raw_data.coordinates, self._raw_data.weights):
            text += "\n" + f"{kpoint[0]} {kpoint[1]} {kpoint[2]}  {weight}"
        return text

    @_documentation.add(
        f"""Read the **k** points data into a dictionary.

Parameters
----------
{_kpoints_opt_source}

Returns
-------
dict
    Contains the coordinates of the **k** points (in crystal units) as
    well as their weights used for integrations. Moreover, some data
    specified in the input file of Vasp are transferred such as the mode
    used to generate the **k** points, the line length (if line mode was
    used), and any labels set for specific points."""
    )
    def _to_dict(self):
        return {
            "mode": self._mode(),
            "line_length": self._line_length(),
            "coordinates": self._raw_data.coordinates[:],
            "weights": self._raw_data.weights[:],
            "labels": self._labels(),
        }

    @_documentation.add(
        f"""Get the number of points per line in the Brillouin zone.

Parameters
----------
{_kpoints_opt_source}

Returns
-------
int
    The number of points used to sample a single line."""
    )
    def _line_length(self):
        if self._mode() == "line":
            return self._raw_data.number
        return len(self._raw_data.coordinates)

    @_documentation.add(
        f"""Get the number of lines in the Brillouin zone.

Parameters
----------
{_kpoints_opt_source}

Returns
-------
int
    The number of lines the band structure contains. For regular meshes this is set to 1."""
    )
    def _number_lines(self):
        return len(self._raw_data.coordinates) // self._line_length()

    @_documentation.add(
        f"""Convert the coordinates of the **k** points into a one dimensional array

For every line in the Brillouin zone, the distance between each **k** point
and the start of the line is calculated. Then the distances of different
lines are concatenated into a single list. This routine is mostly useful
to plot data along high-symmetry lines like band structures.

Parameters
----------
{_kpoints_opt_source}

Returns
-------
np.ndarray
    A reduction of the **k** points onto a one-dimensional array based
    on the distance between the points."""
    )
    def _distances(self):
        cell = self._raw_data.cell.lattice_vectors[-1]
        cartesian_kpoints = np.linalg.solve(cell, self._raw_data.coordinates[:].T).T
        kpoint_lines = np.split(cartesian_kpoints, self._number_lines())
        kpoint_norms = [_line_distances(line) for line in kpoint_lines]
        concatenate_distances = lambda current, addition: (
            np.concatenate((current, addition + current[-1]))
        )
        return functools.reduce(concatenate_distances, kpoint_norms)

    @_documentation.add(
        f"""Get the **k**-point generation mode specified in the Vasp input file

Parameters
----------
{_kpoints_opt_source}

Returns
-------
str
    A string representing which mode was used to setup the k-points."""
    )
    def _mode(self):
        mode = _convert.text_to_string(self._raw_data.mode).strip() or "# empty string"
        first_char = mode[0].lower()
        if first_char == "a":
            return "automatic"
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

    @_documentation.add(
        f"""Get any labels given in the input file for specific **k** points.

Parameters
----------
{_kpoints_opt_source}

Returns
-------
list[str]
    A list of all the k-points explicitly named in the file or the coordinates of the
    band edges if no name was provided."""
    )
    def _labels(self):
        if self._raw_data.label_indices is not None:
            return self._labels_from_file()
        elif self._mode() == "line":
            return self._labels_at_band_edges()
        else:
            return None

    def _labels_from_file(self):
        labels = [""] * len(self._raw_data.coordinates)
        for label, index in zip(self._raw_data.labels, self._raw_indices()):
            labels[index] = _convert.text_to_string(label.strip())
        return labels

    def _raw_indices(self):
        indices = np.array(self._raw_data.label_indices)
        if self._mode() == "line":
            line_length = self._line_length()
            return line_length * (indices // 2) - (indices + 1) % 2
        else:
            return indices - 1  # convert from Fortran to Python indices

    def _labels_at_band_edges(self):
        line_length = self._line_length()
        band_edge = lambda index: not (0 < index % line_length < line_length - 1)
        return [
            _kpoint_label(kpoint) if band_edge(index) else ""
            for index, kpoint in enumerate(self._raw_data.coordinates)
        ]


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
