from py4vasp.data import _util
from py4vasp.data._base import DataBase, RefinementDescriptor
import py4vasp.exceptions as exception
import functools
from fractions import Fraction
import numpy as np


class Kpoints(DataBase):
    """The **k** points used in the Vasp calculation.

    This class provides utility functionality to extract information about the
    **k** points used by Vasp. As such it is mostly used as a helper class for
    other postprocessing classes to extract the required information, e.g., to
    generate a band structure.

    Parameters
    ----------
    raw_kpoints : RawKpoints
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


def _to_string(raw_kpoints):
    text = f"""k-points
{len(raw_kpoints.coordinates)}
reciprocal"""
    for kpoint, weight in zip(raw_kpoints.coordinates, raw_kpoints.weights):
        text += "\n" + f"{kpoint[0]} {kpoint[1]} {kpoint[2]}  {weight}"
    return text


def _to_dict(raw_kpoints):
    """Read the **k** points data into a dictionary.

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
        "mode": _mode(raw_kpoints),
        "line_length": _line_length(raw_kpoints),
        "coordinates": raw_kpoints.coordinates[:],
        "weights": raw_kpoints.weights[:],
        "labels": _labels(raw_kpoints),
    }


def _line_length(raw_kpoints):
    "Get the number of points per line in the Brillouin zone."
    if _mode(raw_kpoints) == "line":
        return raw_kpoints.number
    return len(raw_kpoints.coordinates)


def _number_lines(raw_kpoints):
    "Get the number of lines in the Brillouin zone."
    return len(raw_kpoints.coordinates) // _line_length(raw_kpoints)


def _distances(raw_kpoints):
    """Convert the coordinates of the **k** points into a one dimensional array

    For every line in the Brillouin zone, the distance between each **k** point
    and the start of the line is calculated. Then the distances of different
    lines are concatenated into a single list. This routine is mostly useful
    to plot data along high-symmetry lines like band structures.

    Returns
    -------
    np.ndarray
        A reduction of the **k** points onto a one-dimensional array based
        on the distance between the points.
    """
    cell = raw_kpoints.cell.lattice_vectors * raw_kpoints.cell.scale
    cartesian_kpoints = np.linalg.solve(cell, raw_kpoints.coordinates[:].T).T
    kpoint_lines = np.split(cartesian_kpoints, _number_lines(raw_kpoints))
    kpoint_norms = [np.linalg.norm(line - line[0], axis=1) for line in kpoint_lines]
    concatenate_distances = lambda current, addition: (
        np.concatenate((current, addition + current[-1]))
    )
    return functools.reduce(concatenate_distances, kpoint_norms)


def _mode(raw_kpoints):
    "Get the **k**-point generation mode specified in the Vasp input file"
    mode = _util.decode_if_possible(raw_kpoints.mode).strip() or "# empty string"
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


def _labels(raw_kpoints):
    "Get any labels given in the input file for specific **k** points."
    if raw_kpoints.label_indices is not None:
        return _labels_from_file(raw_kpoints)
    elif _mode(raw_kpoints) == "line":
        return _labels_at_band_edges(raw_kpoints)
    else:
        return None


def _labels_from_file(raw_kpoints):
    labels = [""] * len(raw_kpoints.coordinates)
    for label, index in zip(raw_kpoints.labels, _raw_indices(raw_kpoints)):
        labels[index] = _util.decode_if_possible(label.strip())
    return labels


def _raw_indices(raw_kpoints):
    indices = np.array(raw_kpoints.label_indices)
    if _mode(raw_kpoints) == "line":
        line_length = _line_length(raw_kpoints)
        return line_length * (indices // 2) - (indices + 1) % 2
    else:
        return indices - 1  # convert from Fortran to Python indices


def _labels_at_band_edges(raw_kpoints):
    line_length = _line_length(raw_kpoints)
    band_edge = lambda index: not (0 < index % line_length < line_length - 1)
    return [
        _kpoint_label(kpoint) if band_edge(index) else ""
        for index, kpoint in enumerate(raw_kpoints.coordinates)
    ]


def _kpoint_label(kpoint):
    fractions = [_to_latex(coordinate) for coordinate in kpoint]
    return f"$[{fractions[0]} {fractions[1]} {fractions[2]}]$"


def _to_latex(float):
    fraction = Fraction.from_float(float).limit_denominator()
    if fraction.denominator == 1:
        return str(fraction.numerator)
    else:
        return f"\\frac{{{fraction.numerator}}}{{{fraction.denominator}}}"
