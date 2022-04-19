# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from typing import NamedTuple, Iterable, Union
import numpy as np
import re
from .topology import Topology
from py4vasp.data._base import DataBase, RefinementDescriptor
from py4vasp.data._selection import Selection as _Selection
from py4vasp.data.kpoint import _kpoints_opt_source
import py4vasp.exceptions as exception
import py4vasp._util.convert as _convert
import py4vasp._util.documentation as _documentation
import py4vasp._util.reader as _reader
import py4vasp._util.sanity_check as _check
import py4vasp._util.selection as _selection


_spin_not_set = "not set"
_range = re.compile(r"^(\d+)" + re.escape(_selection.range_separator) + r"(\d+)$")

_selection_doc = r"""
selection : str
    A string specifying the projection of the orbitals. There are three distinct
    possibilities:

    -   To specify the **atom**, you can either use its element name (Si, Al, ...)
        or its index as given in the input file (1, 2, ...). For the latter
        option it is also possible to specify ranges (e.g. 1:4).
    -   To select a particular **orbital** you can give a string (s, px, dxz, ...)
        or select multiple orbitals by their angular momentum (s, p, d, f).
    -   For the **spin**, you have the options up, down, or total.

    You separate multiple selections by commas or whitespace and can nest them using
    parenthesis, e.g. `Sr(s, p)` or `s(up), p(down)`. The order of the selections
    does not matter, but it is case sensitive to distinguish p (angular momentum
    l = 1) from P (phosphorus).
""".strip()


def _selection_examples(instance_name, function_name):
    return f"""Examples
--------
Select the p orbitals of the first atom in the POSCAR file:

>>> calc.{instance_name}.{function_name}(selection="1(p)")

Select the d orbitals of Mn, Co, and Fe:

>>> calc.{instance_name}.{function_name}("d(Mn, Co, Fe)")

Select the spin-up contribution of the first three atoms combined

>>> calc.{instance_name}.{function_name}("up(1{_selection.range_separator}3)")
"""


class Projector(DataBase):
    """The projectors used for atom and orbital resolved quantities.

    This is a common class used by all quantities that contains some projected
    quantity, e.g., the electronic band structure and the DOS. It provides
    utility functionality to access specific indices of the projected arrays
    based on a simple mini language specifying the atom or orbital names.

    Parameters
    ----------
    raw_proj : RawProjector
        Dataclass containing data about the elements, the orbitals, and the spin
        for which projectors are available.
    """

    class Index(NamedTuple):
        "Helper class specifying which atom, orbital, and spin are selected."
        atom: Union[str, _Selection]
        "Label of the atom or a Selection object to read the corresponding data."
        orbital: Union[str, _Selection]
        "Label of the orbital or a Selection object to read the corresponding data."
        spin: Union[str, _Selection]
        "Label of the spin component or a Selection object to read the corresponding data."

    _missing_data_message = "No projectors found, please verify the LORBIT tag is set."

    read = RefinementDescriptor("_to_dict")
    to_dict = RefinementDescriptor("_to_dict")
    select = RefinementDescriptor("_select")
    parse_selection = RefinementDescriptor("_parse_selection")
    __str__ = RefinementDescriptor("_to_string")

    def _to_string(self):
        return f"""projectors:
    atoms: {", ".join(self._topology().ion_types())}
    orbitals: {", ".join(self._orbital_types())}"""

    @_documentation.add(
        f"""Read the selected data from an array and store it in a dictionary.

Parameters
----------
{_selection_doc}
projections : np.ndarray or None
    Array containing projected data.
{_kpoints_opt_source}

Returns
-------
dict
    Dictionary where the label of the selection is linked to a particular
    column of the array. If a particular selection includes multiple indices
    these elements are added. If the projections are not present, the relevant
    indices are returned."""
    )
    def _to_dict(self, selection=None, projections=None):
        if selection is None:
            return {}
        error_message = "Projector selection must be a string."
        _check.raise_error_if_not_string(selection, error_message)
        if projections is None:
            return self._get_indices(selection)
        projections = _Projections(projections)
        return self._read_elements(selection, projections)

    @_documentation.add(
        f"""Map selection strings onto corresponding Selection objects.

With the selection strings, you specify which atom, orbital, and spin component
you are interested in.

Parameters
----------
atom : str
    Element name or index of the atom in the input file of Vasp. If a
    range is specified (e.g. 1{_selection.range_separator}3) a pointer to
    multiple indices will be created.
orbital : str
    Character identifying the angular momentum of the orbital. You may
    select a specific one (e.g. px) or all of the same character (e.g. d).
spin : str
    Select "up" or "down" for a specific spin component or "total" for
    the sum of both.
{_kpoints_opt_source}


Returns
-------
Index
    Indices to access the selected projection from an array and an
    associated label."""
    )
    def _select(
        self,
        atom=_selection.all,
        orbital=_selection.all,
        spin=_selection.all,
    ):
        dicts = self._init_dicts()
        _raise_error_if_not_found_in_dict(orbital, dicts["orbital"])
        _raise_error_if_not_found_in_dict(spin, dicts["spin"])
        return Projector.Index(
            atom=_select_atom(dicts["atom"], atom),
            orbital=dicts["orbital"][orbital],
            spin=dicts["spin"][spin],
        )

    @_documentation.add(
        f"""Generate all possible indices where the projected information is stored.

Given a string specifying which atoms, orbitals, and spin should be selected
an iterable object is created that contains the indices compatible with the
selection.

Parameters
----------
{_selection_doc}
{_kpoints_opt_source}

Yields
------
Iterable[Index]
    Indices of the atom, the orbital and the spin compatible with a specific
    selection."""
    )
    def _parse_selection(self, selection):
        dicts = self._init_dicts()
        default_index = Projector.Index(
            atom=_selection.all,
            orbital=_selection.all,
            spin=_spin_not_set,
        )
        tree = _selection.SelectionTree.from_selection(selection)
        yield from _parse_recursive(dicts, tree, default_index)

    def _topology(self):
        return Topology(self._raw_data.topology)

    def _init_dicts(self):
        return {
            "atom": self._init_atom_dict(),
            "orbital": self._init_orbital_dict(),
            "spin": self._init_spin_dict(),
        }

    def _init_atom_dict(self):
        return self._topology().read()

    def _init_orbital_dict(self):
        num_orbitals = len(self._raw_data.orbital_types)
        all_orbitals = _Selection(indices=slice(num_orbitals))
        orbital_dict = {_selection.all: all_orbitals}
        for i, orbital in enumerate(self._orbital_types()):
            orbital_dict[orbital] = _Selection(indices=slice(i, i + 1), label=orbital)
        if "px" in orbital_dict:
            orbital_dict["p"] = _Selection(indices=slice(1, 4), label="p")
            orbital_dict["d"] = _Selection(indices=slice(4, 9), label="d")
            orbital_dict["f"] = _Selection(indices=slice(9, 16), label="f")
        return orbital_dict

    def _orbital_types(self):
        clean_string = lambda orbital: _convert.text_to_string(orbital).strip()
        for orbital in self._raw_data.orbital_types:
            orbital = clean_string(orbital)
            if orbital == "x2-y2":
                yield "dx2y2"
            else:
                yield orbital

    def _init_spin_dict(self):
        num_spins = self._raw_data.number_spins
        return {
            "polarized": num_spins == 2,
            "up": _Selection(indices=slice(1), label="up"),
            "down": _Selection(indices=slice(1, 2), label="down"),
            "total": _Selection(indices=slice(num_spins), label="total"),
            _selection.all: _Selection(indices=slice(num_spins)),
        }

    def _get_indices(self, selection):
        res = {}
        for select in self._parse_selection(selection):
            atom, orbital, spin = self._select(*select)
            label = _merge_labels([atom.label, orbital.label, spin.label])
            indices = (spin.indices, atom.indices, orbital.indices)
            res[label] = indices
        return res

    def _read_elements(self, selection, projections):
        return {
            label: np.sum(projections[indices], axis=(0, 1, 2))
            for label, indices in self._get_indices(selection).items()
        }


def _merge_labels(labels):
    return "_".join(filter(None, labels))


class _Projections(_reader.Reader):
    def error_message(self, key, err):
        return (
            "Error reading the projections. Please make sure the size of the array "
            f"{self.shape} is compatible with the selected indices. Please also test "
            "if the passed projections allow access by index arrays. "
            "Additionally, you may consider the original error message:\n" + err.args[0]
        )


def _select_atom(atom_dict, atom):
    match = _range.match(atom)
    if match:
        slice_ = _get_slice_from_atom_dict(atom_dict, match)
        return _Selection(indices=slice_, label=atom)
    else:
        _raise_error_if_not_found_in_dict(atom, atom_dict)
        return atom_dict[atom]


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


def _parse_recursive(dicts, tree, current_index):
    for node in tree.nodes:
        new_index = _update_index(dicts, current_index, str(node))
        if len(node.nodes) == 0:
            yield from _setup_spin_indices(new_index, dicts["spin"]["polarized"])
        else:
            yield from _parse_recursive(dicts, node, new_index)


def _update_index(dicts, index, part):
    part = part.strip()
    if part == _selection.all:
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
        yield index._replace(spin=_selection.all)
    else:
        for key in ("up", "down"):
            yield index._replace(spin=key)


class _NoProjectorAvailable:
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
        return _NoProjectorAvailable()
    else:
        return Projector(projectors)
