from py4vasp.data import _util
from py4vasp.exceptions import RefinementException
import functools
import numpy as np


class Kpoints:
    def __init__(self, raw_kpoints):
        self._raw = raw_kpoints

    def read(self):
        return {
            "mode": self.mode(),
            "line_length": self.line_length(),
            "coordinates": self._raw.coordinates[:],
            "weights": self._raw.weights[:],
            "labels": self.labels(),
        }

    def line_length(self):
        if self.mode() == "line":
            return self._raw.number
        return len(self._raw.coordinates)

    def number_lines(self):
        return len(self._raw.coordinates) // self.line_length()

    def distances(self):
        cell = self._raw.cell.lattice_vectors * self._raw.cell.scale
        cartesian_kpoints = np.linalg.solve(cell, self._raw.coordinates[:].T).T
        kpoint_lines = np.split(cartesian_kpoints, self.number_lines())
        kpoint_norms = [np.linalg.norm(line - line[0], axis=1) for line in kpoint_lines]
        concatenate_distances = lambda current, addition: (
            np.concatenate((current, addition + current[-1]))
        )
        return functools.reduce(concatenate_distances, kpoint_norms)

    def mode(self):
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
