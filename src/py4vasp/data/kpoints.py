from py4vasp.data import _util
from py4vasp.exceptions import RefinementException
import functools
import numpy as np


@_util.add_wrappers
class Kpoints:
    """ The **k** points used in the Vasp calculation.

    This class provides utility functionality to extract information about the
    **k** points used by Vasp. As such it is mostly used as a helper class for
    other postprocessing classes to extract the required information, e.g., to
    generate a band structure.

    Parameters
    ----------
    raw_kpoints : raw.Kpoints
        Dataclass containing the raw **k**-points data used in the calculation.
    """

    def __init__(self, raw_kpoints):
        self._raw = raw_kpoints
        self._distances = None

    def to_dict(self):
        """ Read the **k** points data into a dictionary.

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
            "coordinates": self._raw.coordinates[:],
            "weights": self._raw.weights[:],
            "labels": self.labels(),
        }

    def line_length(self):
        "Get the number of points per line in the Brillouin zone."
        if self.mode() == "line":
            return self._raw.number
        return len(self._raw.coordinates)

    def number_lines(self):
        "Get the number of lines in the Brillouin zone."
        return len(self._raw.coordinates) // self.line_length()

    def distances(self):
        """ Convert the coordinates of the **k** points into a one dimensional array

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
        if self._distances is not None:
            return self._distances
        cell = self._raw.cell.lattice_vectors * self._raw.cell.scale
        cartesian_kpoints = np.linalg.solve(cell, self._raw.coordinates[:].T).T
        kpoint_lines = np.split(cartesian_kpoints, self.number_lines())
        kpoint_norms = [np.linalg.norm(line - line[0], axis=1) for line in kpoint_lines]
        concatenate_distances = lambda current, addition: (
            np.concatenate((current, addition + current[-1]))
        )
        self._distances = functools.reduce(concatenate_distances, kpoint_norms)
        return self._distances

    def mode(self):
        "Get the **k**-point generation mode specified in the Vasp input file"
        mode = _util.decode_if_possible(self._raw.mode).strip() or "# empty string"
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
            raise RefinementException(
                "Could not understand the mode '{}' ".format(mode)
                + "when refining the raw kpoints data."
            )

    def labels(self):
        "Get any labels given in the input file for specific **k** points."
        if self._raw.labels is None or self._raw.label_indices is None:
            return None
        labels = [""] * len(self._raw.coordinates)
        use_line_mode = self.mode() == "line"
        for label, index in zip(self._raw.labels, self._raw.label_indices):
            label = _util.decode_if_possible(label.strip())
            if use_line_mode:
                index = self.line_length() * (index // 2) + index % 2
            index -= 1  # convert from Fortran to Python
            labels[index] = label
        return labels
