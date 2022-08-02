# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import re
from typing import NamedTuple, Union
from py4vasp import data
from py4vasp.data import _base, _export
from py4vasp.data._selection import Selection as _Selection
from py4vasp._util import convert as _convert, selection as _selection
import py4vasp._third_party.graph as _graph

_range = re.compile(r"^(\d+)" + re.escape(_selection.range_separator) + r"(\d+)$")


class PhononBand(_base.Refinery, _export.Image):
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
    def __str__(self):
        return f"""phonon band data:
    {self._raw_data.dispersion.eigenvalues.shape[0]} q-points
    {self._raw_data.dispersion.eigenvalues.shape[1]} modes
    {self._topology}"""

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
            ylabel="ω (THz)",
        )

    @_base.data_access
    def to_plotly(self, selection=None, width=1.0):
        return self.plot(selection, width).to_plotly()

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
        result = []
        for index in self._parse_selection(dicts, tree):
            label, selection = _get_selection(dicts, index)
            result.append(self._fat_band(band, label, selection, width))
        return result

    def _init_dicts(self):
        return {
            "atom": self._init_atom_dict(),
            "direction": self._init_direction_dict(),
        }

    def _fat_band(self, band, label, selection, width):
        selected = band["modes"][
            :, :, selection.atom.indices, selection.direction.indices
        ]
        return _graph.Series(
            x=band["qpoint_distances"],
            y=band["bands"].T,
            name=label,
            width=width * np.sum(np.abs(selected), axis=(2, 3)).T,
        )

    def _parse_selection(self, dicts, tree):
        default_index = PhononBand.Index(atom=_selection.all, direction=_selection.all)
        yield from _parse_recursive(dicts, tree, default_index)

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
    return label, PhononBand.Index(selection_atom, selection_direction)


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
