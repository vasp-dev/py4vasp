from __future__ import annotations
from typing import NamedTuple, Iterable, Union
from dataclasses import dataclass
import re
import numpy as np
from .topology import Topology
from py4vasp.data import _util
from py4vasp.data._base import DataBase, RefinementDescriptor
import py4vasp.exceptions as exception

_selection_doc = r"""
selection : str
    A string specifying the projection of the orbitals. There are three distinct
    possibilities:

    -   To specify the **atom**, you can either use its element name (Si, Al, ...)
        or its index as given in the input file (1, 2, ...). For the latter
        option it is also possible to specify ranges (e.g. 1-4).
    -   To select a particular **orbital** you can give a string (s, px, dxz, ...)
        or select multiple orbitals by their angular momentum (s, p, d, f).
    -   For the **spin**, you have the options up, down, or total.

    For all of these options a wildcard \* exists, which selects all elements. You
    separate multiple selections by commas or whitespace and can nest them using
    parenthesis, e.g. `Sr(s, p)` or `s(up), p(down)`. The order of the selections
    does not matter, but is is case sensitive to distinguish p (angular momentum
    l = 1) from P (phosphorus).
""".strip()


class Projectors(DataBase):
    """The projectors used for atom and orbital resolved quantities.

    This is a common class used by all quantities that contains some projected
    quantity, e.g., the electronic band structure and the DOS. It provides
    utility functionality to access specific indices of the projected arrays
    based on a simple mini language specifying the atom or orbital names.

    Parameters
    ----------
    raw_proj : RawProjectors
        Dataclass containing data about the elements, the orbitals, and the spin
        for which projectors are available.
    """

    class Index(NamedTuple):
        "Helper class specifying which atom, orbital, and spin are selected."
        atom: Union[str, Selection]
        "Label of the atom or a Selection object to read the corresponding data."
        orbital: Union[str, Selection]
        "Label of the orbital or a Selection object to read the corresponding data."
        spin: Union[str, Selection]
        "Label of the spin component or a Selection object to read the corresponding data."

    _missing_data_message = "No projectors found, please verify the LORBIT tag is set."

    read = RefinementDescriptor("_to_dict")
    to_dict = RefinementDescriptor("_to_dict")
    select = RefinementDescriptor("_select")
    parse_selection = RefinementDescriptor("_parse_selection")
    __str__ = RefinementDescriptor("_to_string")


def _to_string(raw_proj):
    return f"""projectors:
    atoms: {", ".join(Topology(raw_proj.topology).ion_types())}
    orbitals: {", ".join(_orbital_types(raw_proj))}"""


@_util.add_doc(
    f"""Read the selected data from an array and store it in a dictionary.

Parameters
----------
{_selection_doc}
projections : np.ndarray or None
    Array containing projected data.

Returns
-------
dict
    Dictionary where the label of the selection is linked to a particular
    column of the array. If a particular selection includes multiple indices
    these elements are added. If the projections are not present, the relevant
    indices are returned."""
)
def _to_dict(raw_proj, selection=None, projections=None):
    if selection is None:
        return {}
    error_message = "Projector selection must be a string."
    _util.raise_error_if_not_string(selection, error_message)
    if projections is None:
        return _get_indices(raw_proj, selection)
    projections = _Projections(projections)
    return _read_elements(raw_proj, selection, projections)


def _get_indices(raw_proj, selection):
    res = {}
    for select in _parse_selection(raw_proj, selection):
        atom, orbital, spin = _select(raw_proj, *select)
        label = _merge_labels([atom.label, orbital.label, spin.label])
        indices = (spin.indices, atom.indices, orbital.indices)
        res[label] = indices
    return res


def _merge_labels(labels):
    return "_".join(filter(None, labels))


class _Projections(_util.Reader):
    def error_message(self, key, err):
        return (
            "Error reading the projections. Please make sure the size of the array "
            f"{self.shape} is compatible with the selected indices. Please also test "
            "if the passed projections allow access by index arrays. "
            "Additionally, you may consider the original error message:\n" + err.args[0]
        )


def _read_elements(raw_proj, selection, projections):
    return {
        label: np.sum(projections[indices], axis=(0, 1, 2))
        for label, indices in _get_indices(raw_proj, selection).items()
    }


def _select(
    raw_proj,
    atom=_util.default_selection,
    orbital=_util.default_selection,
    spin=_util.default_selection,
):
    """Map selection strings onto corresponding Selection objects.

    With the selection strings, you specify which atom, orbital, and spin component
    you are interested in. *Note* that for all parameters you can pass "*" to
    default to all (atoms, orbitals, or spins).

    Parameters
    ----------
    atom : str
        Element name or index of the atom in the input file of Vasp. If a
        range is specified (e.g. 1-3) a pointer to multiple indices will be
        created.
    orbital : str
        Character identifying the angular momentum of the orbital. You may
        select a specific one (e.g. px) or all of the same character (e.g. d).
    spin : str
        Select "up" or "down" for a specific spin component or "total" for
        the sum of both.


    Returns
    -------
    Index
        Indices to access the selected projection from an array and an
        associated label.
    """
    dicts = _init_dicts(raw_proj)
    _raise_error_if_not_found_in_dict(orbital, dicts["orbital"])
    _raise_error_if_not_found_in_dict(spin, dicts["spin"])
    return Projectors.Index(
        atom=_select_atom(dicts["atom"], atom),
        orbital=dicts["orbital"][orbital],
        spin=dicts["spin"][spin],
    )


@_util.add_doc(
    f"""Generate all possible indices where the projected information is stored.

Given a string specifying which atoms, orbitals, and spin should be selected
an iterable object is created that contains the indices compatible with the
selection.

Parameters
----------
{_selection_doc}

Yields
------
Iterable[Index]
    Indices of the atom, the orbital and the spin compatible with a specific
    selection."""
)
def _parse_selection(raw_proj, selection):
    dicts = _init_dicts(raw_proj)
    default_index = Projectors.Index(
        atom=_util.default_selection,
        orbital=_util.default_selection,
        spin=_spin_not_set,
    )
    yield from _parse_recursive(dicts, selection, default_index)


_spin_not_set = "not set"


def _init_dicts(raw_proj):
    return {
        "atom": _init_atom_dict(raw_proj),
        "orbital": _init_orbital_dict(raw_proj),
        "spin": _init_spin_dict(raw_proj),
    }


def _init_atom_dict(raw_proj):
    return Topology(raw_proj.topology).read()


def _init_orbital_dict(raw_proj):
    num_orbitals = len(raw_proj.orbital_types)
    all_orbitals = _util.Selection(indices=slice(num_orbitals))
    orbital_dict = {_util.default_selection: all_orbitals}
    for i, orbital in enumerate(_orbital_types(raw_proj)):
        orbital_dict[orbital] = _util.Selection(indices=(i,), label=orbital)
    if "px" in orbital_dict:
        orbital_dict["p"] = _util.Selection(indices=slice(1, 4), label="p")
        orbital_dict["d"] = _util.Selection(indices=slice(4, 9), label="d")
        orbital_dict["f"] = _util.Selection(indices=slice(9, 16), label="f")
    return orbital_dict


def _orbital_types(raw_proj):
    clean_string = lambda ion_type: _util.decode_if_possible(ion_type).strip()
    return (clean_string(orbital) for orbital in raw_proj.orbital_types)


def _init_spin_dict(raw_proj):
    num_spins = raw_proj.number_spins
    return {
        "polarized": num_spins == 2,
        "up": _util.Selection(indices=slice(1), label="up"),
        "down": _util.Selection(indices=slice(1, 2), label="down"),
        "total": _util.Selection(indices=slice(num_spins), label="total"),
        _util.default_selection: _util.Selection(indices=slice(num_spins)),
    }


def _select_atom(atom_dict, atom):
    match = _range.match(atom)
    if match:
        slice_ = _get_slice_from_atom_dict(atom_dict, match)
        return _util.Selection(indices=slice_, label=atom)
    else:
        _raise_error_if_not_found_in_dict(atom, atom_dict)
        return atom_dict[atom]


_range_separator = "-"
_range = re.compile(r"^(\d+)" + re.escape(_range_separator) + r"(\d+)$")


def _get_slice_from_atom_dict(atom_dict, match):
    _raise_error_if_not_found_in_dict(match.groups()[0], atom_dict)
    _raise_error_if_not_found_in_dict(match.groups()[1], atom_dict)
    lower = atom_dict[match.groups()[0]].indices.start
    upper = atom_dict[match.groups()[1]].indices.start
    return slice(lower, upper + 1)


def _raise_error_if_not_found_in_dict(selection, dict_):
    if selection not in dict_:
        raise exception.IncorrectUsage(
            f"Could not find {selection} in projectors. Please check the spelling. "
            f"The available selection are one of {', '.join(dict_)}."
        )


def _parse_recursive(dicts, selection, current_index):
    for part, specification in _split_into_parts(selection):
        new_index = _update_index(dicts, current_index, part)
        if specification == "":
            yield from _setup_spin_indices(new_index, dicts["spin"]["polarized"])
        else:
            yield from _parse_recursive(dicts, specification, new_index)


def _update_index(dicts, index, part):
    part = part.strip()
    if part == _util.default_selection:
        pass
    elif part in dicts["atom"] or _range.match(part):
        index = index._replace(atom=part)
    elif part in dicts["orbital"]:
        index = index._replace(orbital=part)
    elif part in dicts["spin"]:
        index = index._replace(spin=part)
    else:
        raise exception.IncorrectUsage(
            f"Could not find {part} in the list of projectors. Please check "
            "if everything is spelled correctly. Notice that the selection is case "
            "sensitive so that 's' (orbital) can be distinguished from 'S' (sulfur)."
        )
    return index


def _setup_spin_indices(index, spin_polarized):
    if index.spin != _spin_not_set:
        yield index
    elif not spin_polarized:
        yield index._replace(spin=_util.default_selection)
    else:
        for key in ("up", "down"):
            yield index._replace(spin=key)


class _NoProjectorsAvailable:
    def read(self, selection, projections):
        if selection is not None:
            raise exception.IncorrectUsage(
                "Projectors are not available, rerun Vasp setting LORBIT >= 10."
            )
        return {}

    def __repr__(self):
        return ""


def _projectors_or_dummy(projectors):
    if projectors is None:
        return _NoProjectorsAvailable()
    else:
        return Projectors(projectors)


def _split_into_parts(selection):
    selection = _cleanup_whitespace(selection)
    state = _State()
    for char in selection + _seperators[0]:  # make sure selection contains termination
        state = _update_state(state, char)
        if state.complete:
            yield state.part, state.specification


_seperators = (" ", ",")


def _cleanup_whitespace(selection):
    selection = _whitespace_begin_spec.sub(_begin_spec, selection)
    selection = _whitespace_end_spec.sub(_end_spec + _seperators[0], selection)
    return _whitespace_range.sub(_range_separator, selection)


_begin_spec = "("
_end_spec = ")"
_whitespace_begin_spec = re.compile(r"\s*" + re.escape(_begin_spec) + r"\s*")
_whitespace_end_spec = re.compile(r"\s*" + re.escape(_end_spec) + r"\s*")
_whitespace_range = re.compile(r"\s*" + re.escape(_range_separator) + r"\s*")


@dataclass
class _State:
    level: int = 0
    part: str = ""
    specification: str = ""
    complete: bool = False


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
