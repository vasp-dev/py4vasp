from py4vasp.data import _util
from py4vasp.data._base import DataBase, RefinementDescriptor
import py4vasp.exceptions as exception
import functools
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
    fortran_indices = _raw_indices(raw_kpoints)
    if fortran_indices is None:
        return None
    python_indices = np.array(fortran_indices) - 1
    labels = [""] * len(raw_kpoints.coordinates)
    for label, index in zip(_raw_labels(raw_kpoints), python_indices):
        labels[index] = label
    return labels


def _raw_labels(raw_kpoints):
    if raw_kpoints.labels is not None:
        return (_util.decode_if_possible(label.strip()) for label in raw_kpoints.labels)
    else:
        distances = [f"{distance:.2g}" for distance in _distances(raw_kpoints)]
        return distances[:: _line_length(raw_kpoints)] + [distances[-1]]


def _raw_indices(raw_kpoints):
    indices = raw_kpoints.label_indices
    line_length = _line_length(raw_kpoints)
    if _mode(raw_kpoints) != "line":
        return indices
    elif indices is not None:
        indices = np.array(indices)
        return line_length * (indices // 2) + indices % 2
    else:
        indices = np.arange(len(raw_kpoints.coordinates) + 1, step=line_length)
        indices[:-1] += 1  # convert to Fortran index
        return indices
