from py4vasp.data import _util
from py4vasp.exceptions import RefinementException


class Kpoints:
    def __init__(self, raw_kpoints):
        self._raw = raw_kpoints

    def read(self):
        return {
            "mode": self._mode(),
            "coordinates": self._raw.coordinates[:],
            "weights": self._raw.weights[:],
            "labels": None,
        }

    def line_length(self):
        if self._mode() == "line":
            return self._raw.number
        return len(self._raw.coordinates)

    def number_lines(self):
        return len(self._raw.coordinates) // self.line_length()

    def _mode(self):
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
