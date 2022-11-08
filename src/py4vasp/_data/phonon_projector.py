# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import annotations

import re
import typing

from py4vasp._data.selection import Selection
from py4vasp._util import select

_range = re.compile(r"^(\d+)" + re.escape(select.range_separator) + r"(\d+)$")
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


class PhononProjector:
    """Facilitate selecting projections on particular atoms and directions

    Use this class to project the phonon density of states or phonon band structure
    on particular atoms or directions.

    Parameters
    ----------
    topology - data.Topology
        Defines the kind and number of atoms.
    """

    class Index(typing.NamedTuple):
        "Helper class specifying which atom and direction are selected."
        atom: Union[str, Selection] = select.all
        "Label of the atom or a Selection object to read the corresponding data."
        direction: Union[str, Selection] = select.all
        "Label of the direction or a Selection object to read the corresponding data."

    def __init__(self, topology):
        self._atom_dict = topology.read()
        self._direction_dict = {
            select.all: Selection(slice(0, 3)),
            "x": Selection(slice(0, 1), "x"),
            "y": Selection(slice(1, 2), "y"),
            "z": Selection(slice(2, 3), "z"),
        }
        self.modes = 3 * topology.number_atoms()

    def parse_selection(self, selection):
        """Split a string into the relevant substrings defining the selection.

        Parameters
        ----------
        selection - str
            A user provided string selecting atoms by their name or index and/or
            cartesian directions x, y, z.
        """
        tree = select.Tree.from_selection(selection)
        default_index = PhononProjector.Index()
        yield from self._parse_recursive(tree, default_index)

    def _parse_recursive(self, tree, current_index):
        for node in tree.nodes:
            new_index = self._update_index(current_index, str(node))
            if node.nodes:
                yield from self._parse_recursive(node, new_index)
            else:
                yield new_index

    def _update_index(self, index, part):
        part = part.strip()
        if part in self._atom_dict or _range.match(part):
            index = index._replace(atom=part)
        elif part in self._direction_dict:
            index = index._replace(direction=part)
        else:
            pass
        return index

    def select(self, atom, direction):
        """Convert atom and direction selections into slices.

        Parameters
        ----------
        atom - str
            Name of the atom (e.g. Sr) or index of the atom (e.g. 3) or range of indices
            (e.g. 2:5)
        direction - str
            Name of a cartesian direction (x, y, z)
        """
        selection_atom = self._select_atom(atom)
        selection_direction = self._direction_dict[direction]
        label = _merge_labels(selection_atom.label, selection_direction.label)
        return label, PhononProjector.Index(selection_atom, selection_direction)

    def _select_atom(self, atom):
        if match := _range.match(atom):
            slice_ = self._get_slice_from_atom_dict(match)
            return Selection(indices=slice_, label=atom)
        else:
            return self._atom_dict[atom]

    def _get_slice_from_atom_dict(self, match):
        start = self._atom_dict[match.groups()[0]].indices.start
        stop = self._atom_dict[match.groups()[1]].indices.start + 1
        return slice(start, stop)


def _merge_labels(label_atom, label_direction):
    if label_atom and label_direction:
        return f"{label_atom}_{label_direction}"
    else:
        return label_atom or label_direction
