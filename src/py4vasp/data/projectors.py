from __future__ import annotations
from typing import NamedTuple, Iterable, Union
from dataclasses import dataclass
import functools
import itertools
import re
import numpy as np
from py4vasp.data import _util
from py4vasp.exceptions import UsageException


_default = "*"
_spin_not_set = "not set"
_begin_spec = "("
_end_spec = ")"
_seperators = (" ", ",")
_range_separator = "-"
_range = re.compile(r"^(\d+)" + re.escape(_range_separator) + "(\d+)$")
_whitespace_begin_spec = re.compile(r"\s*" + re.escape(_begin_spec) + r"\s*")
_whitespace_end_spec = re.compile(r"\s*" + re.escape(_end_spec) + r"\s*")
_whitespace_range = re.compile(r"\s*" + re.escape(_range_separator) + r"\s*")


@dataclass
class _State:
    level: int = 0
    part: str = ""
    specification: str = ""
    complete: bool = False


def _split_into_parts(selection):
    selection = _cleanup_whitespace(selection)
    state = _State()
    for char in selection + _seperators[0]:  # make sure selection contains termination
        state = _update_state(state, char)
        if state.complete:
            yield state.part, state.specification


def _cleanup_whitespace(selection):
    selection = _whitespace_begin_spec.sub(_begin_spec, selection)
    selection = _whitespace_end_spec.sub(_end_spec + _seperators[0], selection)
    return _whitespace_range.sub(_range_separator, selection)


def _update_state(state, char):
    state.level = _update_level(state, char)
    state.part = _update_part(state, char)
    state.specification = _update_specification(state, char)
    state.complete = _is_state_complete(state, char)
    return state


def _update_level(state, char):
    return state.level + (char == _begin_spec) - (char == _end_spec)


def _update_part(state, char):
    part = state.part if not state.complete else ""
    char_used = char not in (_end_spec, *_seperators) and state.level == 0
    char = char if char_used else ""
    return part + char


def _update_specification(state, char):
    spec_used = not state.complete and (state.level != 1 or char != _begin_spec)
    spec = state.specification if spec_used else ""
    char_used = spec_used and state.level > 0
    char = char if char_used else ""
    return spec + char


def _is_state_complete(state, char):
    return state.level == 0 and char in _seperators and state.part != ""


class Projectors:
    class Selection(NamedTuple):
        indices: Iterable[int]
        label: str = ""

    class Index(NamedTuple):
        atom: Union[str, Selection]
        orbital: Union[str, Selection]
        spin: Union[str, Selection]

    def __init__(self, raw_proj):
        self._raw = raw_proj
        self._init_atom_dict(raw_proj)
        self._init_orbital_dict(raw_proj)
        self._init_spin_dict(raw_proj)
        self._spin_polarized = raw_proj.number_spins == 2

    @classmethod
    def from_file(cls, file=None):
        return _util.from_file(cls, file, "projectors")

    def _init_atom_dict(self, raw_proj):
        num_atoms = np.sum(raw_proj.number_ion_types)
        all_atoms = self.Selection(indices=range(num_atoms))
        self._atom_dict = {_default: all_atoms}
        start = 0
        for type, number in zip(raw_proj.ion_types, raw_proj.number_ion_types):
            type = str(type, "utf-8").strip()
            _range = range(start, start + number)
            self._atom_dict[type] = self.Selection(indices=_range, label=type)
            for i in _range:
                # create labels like Si_1, Si_2, Si_3 (starting at 1)
                label = type + "_" + str(_range.index(i) + 1)
                self._atom_dict[str(i + 1)] = self.Selection(indices=(i,), label=label)
            start += number

    def _init_orbital_dict(self, raw_proj):
        num_orbitals = len(raw_proj.orbital_types)
        all_orbitals = self.Selection(indices=range(num_orbitals))
        self._orbital_dict = {_default: all_orbitals}
        for i, orbital in enumerate(raw_proj.orbital_types):
            orbital = str(orbital, "utf-8").strip()
            self._orbital_dict[orbital] = self.Selection(indices=(i,), label=orbital)
        if "px" in self._orbital_dict:
            self._orbital_dict["p"] = self.Selection(indices=range(1, 4), label="p")
            self._orbital_dict["d"] = self.Selection(indices=range(4, 9), label="d")
            self._orbital_dict["f"] = self.Selection(indices=range(9, 16), label="f")

    def _init_spin_dict(self, raw_proj):
        num_spins = raw_proj.number_spins
        self._spin_dict = {
            "up": self.Selection(indices=(0,), label="up"),
            "down": self.Selection(indices=(1,), label="down"),
            "total": self.Selection(indices=range(num_spins), label="total"),
            _default: self.Selection(indices=range(num_spins)),
        }

    def select(self, atom=_default, orbital=_default, spin=_default):
        return self.Index(
            atom=self._select_atom(atom),
            orbital=self._orbital_dict[orbital],
            spin=self._spin_dict[spin],
        )

    def _select_atom(self, atom):
        match = _range.match(atom)
        if match:
            lower = self._atom_dict[match.groups()[0]].indices[0]
            upper = self._atom_dict[match.groups()[1]].indices[0]
            return self.Selection(indices=range(lower, upper + 1), label=atom)
        else:
            return self._atom_dict[atom]

    def parse_selection(self, selection):
        default_index = self.Index(atom=_default, orbital=_default, spin=_spin_not_set)
        yield from self._parse_recursive(selection, default_index)

    def _parse_recursive(self, selection, current_index):
        for part, specification in _split_into_parts(selection):
            new_index = self._update_index(current_index, part)
            if specification == "":
                yield from self._setup_spin_indices(new_index)
            else:
                yield from self._parse_recursive(specification, new_index)

    def _update_index(self, index, part):
        part = part.strip()
        if part == _default:
            pass
        elif part in self._atom_dict or _range.match(part):
            index = index._replace(atom=part)
        elif part in self._orbital_dict:
            index = index._replace(orbital=part)
        elif part in self._spin_dict:
            index = index._replace(spin=part)
        else:
            raise KeyError("Could not find " + part + " in the list or projectors.")
        return index

    def _setup_spin_indices(self, index):
        if index.spin != _spin_not_set:
            yield index
        elif not self._spin_polarized:
            yield index._replace(spin=_default)
        else:
            for key in ("up", "down"):
                yield index._replace(spin=key)

    def read(self, selection, projections):
        if selection is None:
            return {}
        return self._read_elements(selection, projections)

    def _read_elements(self, selection, projections):
        res = {}
        for select in self.parse_selection(selection):
            atom, orbital, spin = self.select(*select)
            label = self._merge_labels([atom.label, orbital.label, spin.label])
            orbitals = self._filter_orbitals(orbital.indices, projections.shape[2])
            index = (spin.indices, atom.indices, orbitals)
            res[label] = self._read_element(index, projections)
        return res

    def _merge_labels(self, labels):
        return "_".join(filter(None, labels))

    def _filter_orbitals(self, orbitals, number_orbitals):
        return filter(lambda x: x < number_orbitals, orbitals)

    def _read_element(self, index, projections):
        sum_projections = lambda proj, i: proj + projections[i]
        zeros = np.zeros(projections.shape[3:])
        return functools.reduce(sum_projections, itertools.product(*index), zeros)


class _NoProjectorsAvailable:
    def read(self, selection, projections):
        if selection is not None:
            raise UsageException(
                "Projectors are not available, rerun Vasp setting LORBIT = 10 or 11."
            )
        return {}


def _projectors_or_dummy(projectors):
    if projectors is None:
        return _NoProjectorsAvailable()
    else:
        return Projectors(projectors)
