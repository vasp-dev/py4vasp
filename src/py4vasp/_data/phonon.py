# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import data
from py4vasp._util import select

selection_doc = """\
selection : str
    A string specifying the projection of the phonon modes onto atoms and directions.
    Please specify selections using one of the following:

    -   To specify the **atom**, you can either use its element name (Si, Al, ...)
        or its index as given in the input file (1, 2, ...). For the latter
        option it is also possible to specify ranges (e.g. 1:4).
    -   To select a particular **direction** specify the Cartesian direction (x, y, z).

    You separate multiple selections by commas or whitespace and can nest them using
    parenthesis, e.g. `Sr(x)` or `z(1, 2)`. The order of the selections does not matter,
    but it is case sensitive to distinguish y (Cartesian direction) from Y (yttrium).
"""


class Mixin:
    "Provide functionality common to Phonon classes."

    def _topology(self):
        return data.Topology.from_data(self._raw_data.topology)

    def _init_atom_dict(self):
        return {
            key: value.indices
            for key, value in self._topology().read().items()
            if key != select.all
        }

    def _init_direction_dict(self):
        return {
            "x": slice(0, 1),
            "y": slice(1, 2),
            "z": slice(2, 3),
        }
