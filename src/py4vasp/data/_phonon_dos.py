# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import annotations
import numpy as np
import typing
import re
from py4vasp import data
from py4vasp.data import _base
from py4vasp.data._selection import Selection as _Selection
from py4vasp._util import selection as _selection


_range = re.compile(r"^(\d+)" + re.escape(_selection.range_separator) + r"(\d+)$")


class PhononDos(_base.Refinery):
    """The phonon density of states (DOS).

    You can use this class to extract the phonon DOS data of a VASP
    calculation. The DOS can also be resolved by direction and atom.
    """

    class Index(typing.NamedTuple):
        "Helper class specifying which atom and direction are selected."
        atom: Union[str, _Selection]
        "Label of the atom or a Selection object to read the corresponding data."
        direction: Union[str, _Selection]
        "Label of the direction or a Selection object to read the corresponding data."

    def to_dict(self, selection=None):
        return {
            "energies": self._raw_data.energies[:],
            "total": self._raw_data.dos[:],
            **self._read_data(selection),
        }

    def _read_data(self, selection):
        tree = _selection.Tree.from_selection(selection)
        dicts = self._init_dicts()
        result = {}
        for index in self._parse_selection(dicts, tree):
            label, selection = _get_selection(dicts, index)
            result[label] = self._partial_dos(selection)
            print(result[label].shape)
        return result

    def _parse_selection(self, dicts, tree):
        default_index = PhononDos.Index(atom=_selection.all, direction=_selection.all)
        yield from _parse_recursive(dicts, tree, default_index)

    def _init_dicts(self):
        return {
            "atom": self._init_atom_dict(),
            "direction": self._init_direction_dict(),
        }

    def _init_atom_dict(self):
        return self._topology.read()

    @property
    def _topology(self):
        return data.Topology.from_data(self._raw_data.topology)

    def _init_direction_dict(self):
        return {
            _selection.all: _Selection(slice(0, 3)),
            "x": _Selection(slice(0, 1), "x"),
            "y": _Selection(slice(1, 2), "y"),
            "z": _Selection(slice(2, 3), "z"),
        }

    def _partial_dos(self, selection):
        projections = self._raw_data.projections[
            selection.atom.indices, selection.direction.indices
        ]
        return np.sum(projections, axis=(0, 1))


def _parse_recursive(dicts, tree, current_index):
    for node in tree.nodes:
        new_index = _update_index(dicts, current_index, str(node))
        if node.nodes:
            yield from _parse_recursive(dicts, node, new_index)
        else:
            yield new_index


def _update_index(dicts, index, part):
    part = part.strip()
    if part in dicts["atom"] or _range.match(part):
        index = index._replace(atom=part)
    elif part in dicts["direction"]:
        index = index._replace(direction=part)
    else:
        pass
    return index


def _get_selection(dicts, index):
    selection_atom = _select_atom(dicts["atom"], index.atom)
    selection_direction = dicts["direction"][index.direction]
    label = _merge_labels(selection_atom.label, selection_direction.label)
    return label, PhononDos.Index(selection_atom, selection_direction)


def _select_atom(atom_dict, atom):
    if match := _range.match(atom):
        slice_ = _get_slice_from_atom_dict(atom_dict, match)
        return _Selection(indices=slice_, label=atom)
    else:
        return atom_dict[atom]


def _get_slice_from_atom_dict(atom_dict, match):
    start = atom_dict[match.groups()[0]].indices.start
    stop = atom_dict[match.groups()[1]].indices.start + 1
    return slice(start, stop)


def _merge_labels(label_atom, label_direction):
    if label_atom and label_direction:
        return f"{label_atom}_{label_direction}"
    else:
        return label_atom or label_direction
