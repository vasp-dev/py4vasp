# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import re
from typing import NamedTuple, Union
from py4vasp import data
from py4vasp.data import _base
from py4vasp.data._selection import Selection as _Selection
from py4vasp._util import convert as _convert, selection as _selection
import py4vasp._third_party.graph as _graph

_range = re.compile(r"^(\d+)" + re.escape(_selection.range_separator) + r"(\d+)$")


class PhononBand(_base.Refinery):
    """The phonon band structure.

    Use this to examine the phonon band structure along a high-symmetry path in the
    Brillouin zone. The `to_dict` function allows to extract the raw data to process
    it further."""

    class Index(NamedTuple):
        "Helper class specifying which atom and direction are selected."
        atom: Union[str, _Selection]
        "Label of the atom or a Selection object to read the corresponding data."
        direction: Union[str, _Selection]
        "Label of the direction or a Selection object to read the corresponding data."

    @_base.data_access
    def to_dict(self):
        return {
            "qpoint_distances": self._qpoints.distances(),
            "bands": self._raw_data.dispersion.eigenvalues[:],
            "modes": _convert.to_complex(self._raw_data.eigenvectors[:]),
        }

    @_base.data_access
    def plot(self, selection=None, width=1.0):
        return _graph.Graph(
            series=self._band_structure(selection, width),
            ylabel="Energy (meV)",
        )

    @property
    def _qpoints(self):
        return data.Kpoint.from_data(self._raw_data.dispersion.kpoints)

    def _band_structure(self, selection, width):
        band = self.to_dict()
        tree = _selection.Tree.from_selection(selection)
        if tree.nodes:
            return self._fat_band_structure(band, tree, width)
        else:
            return self._regular_band_structure(band)

    def _regular_band_structure(self, band):
        return [_graph.Series(x=band["qpoint_distances"], y=band["bands"].T)]

    def _fat_band_structure(self, band, tree, width):
        dicts = self._init_dicts()
        return [
            self._fat_band(band, dicts, index, width)
            for index in self._parse_selection(dicts, tree)
        ]

    def _init_dicts(self):
        return {
            "atom": self._init_atom_dict(),
            "direction": self._init_direction_dict(),
        }

    def _fat_band(self, band, dicts, index, width):
        selection = _get_selection(dicts, index)
        print(selection)
        selected = band["modes"][:, selection.indices, :]
        return _graph.Series(
            x=band["qpoint_distances"],
            y=band["bands"].T,
            name=selection.label,
            width=width * np.sum(np.abs(selected), axis=1),
        )

    def _parse_selection(self, dicts, tree):
        default_index = PhononBand.Index(atom=_selection.all, direction=_selection.all)
        yield from _parse_recursive(dicts, tree, default_index)

    def _init_atom_dict(self):
        return data.Topology.from_data(self._raw_data.topology).read()

    def _init_direction_dict(self):
        return {
            _selection.all: _Selection(slice(0, 3)),
            "x": _Selection(slice(0, 1), "x"),
            "y": _Selection(slice(1, 2), "y"),
            "z": _Selection(slice(2, 3), "z"),
        }


def _get_selection(dicts, index):
    selection_atom = _select_atom(dicts["atom"], index.atom)
    selection_direction = dicts["direction"][index.direction]
    indices = _merge_indices(selection_atom.indices, selection_direction.indices)
    label = _merge_labels(selection_atom.label, selection_direction.label)
    return _Selection(indices, label)


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


def _merge_indices(index_atom, index_direction):
    num_direction = 3
    start = index_atom.start * num_direction + index_direction.start
    stop = (index_atom.stop - 1) * num_direction + index_direction.stop
    step = num_direction - index_direction.stop + index_direction.start + 1
    return slice(start, stop, step)


def _merge_labels(label_atom, label_direction):
    if label_atom and label_direction:
        return f"{label_atom}_{label_direction}"
    else:
        return label_atom or label_direction


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
