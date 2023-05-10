# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import re
import warnings
from typing import NamedTuple, Union

import numpy as np

from py4vasp import data, exception
from py4vasp._data import base
from py4vasp._data.selection import Selection
from py4vasp._util import convert, documentation, index, reader, select

selection_doc = """\
selection : str
    A string specifying the projection of the orbitals. There are four distinct
    possibilities:

    -   To specify the **atom**, you can either use its element name (Si, Al, ...)
        or its index as given in the input file (1, 2, ...). For the latter
        option it is also possible to specify ranges (e.g. 1:4).
    -   To select a particular **orbital** you can give a string (s, px, dxz, ...)
        or select multiple orbitals by their angular momentum (s, p, d, f).
    -   For the **spin**, you have the options up, down, or total.
    -   If you used a different **k**-point mesh choose "kpoints_opt" or "kpoints_wan"
        to select them instead of the default mesh specified in the KPOINTS file.

    You separate multiple selections by commas or whitespace and can nest them using
    parenthesis, e.g. `Sr(s, p)` or `s(up), p(down)`. The order of the selections
    does not matter, but it is case sensitive to distinguish p (angular momentum
    l = 1) from P (phosphorus).

    If you are unsure about the specific projections that are available, you can use

    >>> calc.projector.selections()

    to get a list of all available ones.
"""


def selection_examples(instance_name, function_name):
    return f"""Examples
--------
Select the p orbitals of the first atom in the POSCAR file:

>>> calc.{instance_name}.{function_name}(selection="1(p)")

Select the d orbitals of Mn, Co, and Fe:

>>> calc.{instance_name}.{function_name}("d(Mn, Co, Fe)")

Select the spin-up contribution of the first three atoms combined

>>> calc.{instance_name}.{function_name}("up(1{select.range_separator}3)")
"""


_spin_not_set = "not set"
_range = re.compile(r"^(\d+)" + re.escape(select.range_separator) + r"(\d+)$")
_select_all = select.all


class Projector(base.Refinery):
    """The projectors used for atom and orbital resolved quantities.

    This is a common class used by all quantities that contains some projected
    quantity, e.g., the electronic band structure and the DOS. It provides
    utility functionality to access specific indices of the projected arrays
    based on a simple mini language specifying the atom or orbital names.
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

    @base.data_access
    def __str__(self):
        if self._raw_data.orbital_types.is_none():
            return "no projectors"
        return f"""projectors:
    atoms: {", ".join(self._topology().ion_types())}
    orbitals: {", ".join(self._orbital_types())}"""

    @base.data_access
    @documentation.format(selection_doc=selection_doc)
    def to_dict(self, selection=None, projections=None):
        """Read the selected data from an array and store it in a dictionary.

        Parameters
        ----------
        {selection_doc}
        projections : np.ndarray or None
            Array containing projected data.

        Returns
        -------
        dict
            Dictionary where the label of the selection is linked to a particular
            column of the array. If a particular selection includes multiple indices
            these elements are added. If the projections are not present, the relevant
            indices are returned.
        """
        if selection is None:
            return self._init_dicts()
        message = "Calling `Projector.to_dict` with selection is deprecated, please use `Projector.project` instead."
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return self.project(selection, projections)

    @base.data_access
    def project(self, selection, projections):
        if not selection:
            return {}
        self._raise_error_if_orbitals_missing()
        selector = self._make_selector(projections)
        return {
            selector.label(selection): selector[selection]
            for selection in self._parse_selection(selection)
        }

    def _make_selector(self, projections):
        maps = self.to_dict()
        maps = {1: maps["atom"], 2: maps["orbital"], 0: maps["spin"]}
        try:
            return index.Selector(maps, projections)
        except exception._Py4VaspInternalError as error:
            message = f"""Error reading the projections. Please make sure that the passed
                projections has the right format, i.e., the indices correspond to spin,
                atom, and orbital, respectively."""
            raise exception.IncorrectUsage(message) from error

    def _parse_selection(self, selection):
        tree = select.Tree.from_selection(selection)
        for selection in tree.selections():
            if not self._spin_polarized or self._spin_selected(selection):
                yield selection
            else:
                yield from self._add_spin_components(selection)

    def _spin_selected(self, selection):
        return any(
            select.contains(selection, choice) for choice in self._init_spin_dict()
        )

    def _add_spin_components(self, selection):
        yield selection + ("up",)
        yield selection + ("down",)

    @base.data_access
    @documentation.format(separator=select.range_separator)
    def select(
        self,
        atom=_select_all,
        orbital=_select_all,
        spin=_select_all,
    ):
        """Map selection strings onto corresponding Selection objects.

        With the selection strings, you specify which atom, orbital, and spin component
        you are interested in.

        Parameters
        ----------
        atom : str
            Element name or index of the atom in the input file of Vasp. If a
            range is specified (e.g. 1{separator}3) a pointer to
            multiple indices will be created.
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
        dicts = self._init_dicts_old()
        _raise_error_if_not_found_in_dict(orbital, dicts["orbital"])
        _raise_error_if_not_found_in_dict(spin, dicts["spin"])
        return Projector.Index(
            atom=_select_atom(dicts["atom"], atom),
            orbital=dicts["orbital"][orbital],
            spin=dicts["spin"][spin],
        )

    @base.data_access
    @documentation.format(selection_doc=selection_doc)
    def parse_selection(self, selection=_select_all):
        """Generate all possible indices where the projected information is stored.

        Given a string specifying which atoms, orbitals, and spin should be selected
        an iterable object is created that contains the indices compatible with the
        select.

        Parameters
        ----------
        {selection_doc}

        Yields
        ------
        Iterable[Index]
            Indices of the atom, the orbital and the spin compatible with a specific
            selection."""
        dicts = self._init_dicts_old()
        default_index = Projector.Index(
            atom=_select_all,
            orbital=_select_all,
            spin=_spin_not_set,
        )
        tree = select.Tree.from_selection(selection)
        yield from _parse_recursive(dicts, tree, default_index)

    @base.data_access
    def selections(self):
        """Return a dictionary describing what options are available to specify the
        atom, orbital, and spin."""
        dicts = self._init_dicts_old()
        return {
            "atom": _filter_and_sort_keys(dicts["atom"]),
            "orbital": _filter_and_sort_keys(dicts["orbital"]),
            "spin": _filter_and_sort_keys(dicts["spin"]),
        }

    def _topology(self):
        return data.Topology.from_data(self._raw_data.topology)

    def _init_dicts(self):
        if self._raw_data.orbital_types.is_none():
            return {}
        atom_dict = self._init_atom_dict()
        orbital_dict = self._init_orbital_dict()
        spin_dict = self._init_spin_dict()
        return {"atom": atom_dict, "orbital": orbital_dict, "spin": spin_dict}

    def _init_atom_dict(self):
        return {
            key: value.indices
            for key, value in self._topology().read().items()
            if key != _select_all
        }

    def _init_orbital_dict(self):
        orbital_dict = {
            orbital: slice(i, i + 1) for i, orbital in enumerate(self._orbital_types())
        }
        if "px" in orbital_dict:
            orbital_dict["p"] = slice(1, 4)
            orbital_dict["d"] = slice(4, 9)
            orbital_dict["f"] = slice(9, 16)
        return orbital_dict

    def _init_spin_dict(self):
        if not self._spin_polarized:
            return {"total": slice(0, 1)}
        return {"total": slice(0, 2), "up": slice(0, 1), "down": slice(1, 2)}

    @property
    def _spin_polarized(self):
        return self._raw_data.number_spins == 2

    def _init_dicts_old(self):
        return {
            "atom": self._init_atom_dict_old(),
            "orbital": self._init_orbital_dict_old(),
            "spin": self._init_spin_dict_old(),
        }

    def _init_atom_dict_old(self):
        return self._topology().read()

    def _init_orbital_dict_old(self):
        self._raise_error_if_orbitals_missing()
        num_orbitals = len(self._raw_data.orbital_types)
        all_orbitals = Selection(indices=slice(0, num_orbitals))
        orbital_dict = {_select_all: all_orbitals}
        for i, orbital in enumerate(self._orbital_types()):
            orbital_dict[orbital] = Selection(indices=slice(i, i + 1), label=orbital)
        if "px" in orbital_dict:
            orbital_dict["p"] = Selection(indices=slice(1, 4), label="p")
            orbital_dict["d"] = Selection(indices=slice(4, 9), label="d")
            orbital_dict["f"] = Selection(indices=slice(9, 16), label="f")
        return orbital_dict

    def _raise_error_if_orbitals_missing(self):
        if self._raw_data.orbital_types.is_none():
            message = "Projectors are not available, rerun Vasp setting LORBIT >= 10."
            raise exception.IncorrectUsage(message)

    def _orbital_types(self):
        clean_string = lambda orbital: convert.text_to_string(orbital).strip()
        for orbital in self._raw_data.orbital_types:
            orbital = clean_string(orbital)
            if orbital == "x2-y2":
                yield "dx2y2"
            else:
                yield orbital

    def _init_spin_dict_old(self):
        num_spins = self._raw_data.number_spins
        result = {
            "polarized": num_spins == 2,
            "total": Selection(indices=slice(0, num_spins), label="total"),
            _select_all: Selection(indices=slice(0, num_spins)),
        }
        if num_spins == 2:
            result["up"] = Selection(indices=slice(0, 1), label="up")
            result["down"] = Selection(indices=slice(1, 2), label="down")
        return result

    def _get_indices(self, selection):
        res = {}
        for select in self.parse_selection(selection):
            atom, orbital, spin = self.select(*select)
            label = _merge_labels([atom.label, orbital.label, spin.label])
            indices = (spin.indices, atom.indices, orbital.indices)
            res[label] = indices
        return res

    def _read_elements(self, indices, projections):
        projections = _Projections(projections)
        return {
            label: np.sum(projections[indices], axis=(0, 1, 2))
            for label, indices in indices.items()
        }


def _merge_labels(labels):
    return "_".join(filter(None, labels))


class _Projections(reader.Reader):
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
        return Selection(indices=slice_, label=atom)
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
            f"Could not find `{selection}` in projectors. Please check the spelling. "
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
    if part == _select_all:
        pass
    elif part in dicts["atom"] or _range.match(part):
        index = index._replace(atom=part)
    elif part in dicts["orbital"]:
        index = index._replace(orbital=part)
    elif part in dicts["spin"]:
        index = index._replace(spin=part)
    elif part in ["up", "down"]:
        raise exception.IncorrectUsage(
            f"Could not find `{part}` in the list of projectors because there is only "
            "one spin component in the HDF5 file. Please check whether you set ISPIN = 2 "
            "and make sure the calculation already completed."
        )
    else:
        raise exception.IncorrectUsage(
            f"Could not find `{part}` in the list of projectors. Please check "
            "if everything is spelled correctly. Notice that the selection is case "
            "sensitive so that 's' (orbital) can be distinguished from 'S' (sulfur)."
        )
    return index


def _setup_spin_indices(index, spin_polarized):
    if index.spin != _spin_not_set:
        yield index
    elif not spin_polarized:
        yield index._replace(spin=_select_all)
    else:
        for key in ("up", "down"):
            yield index._replace(spin=key)


def _filter_and_sort_keys(keys):
    filtered_keys = filter(lambda key: key not in [_select_all, "polarized"], keys)
    return sorted(filtered_keys, key=_sort_key)


def _sort_key(key):
    spin_keys = ["total", "up", "down"]
    orbital_keys = ["s", "p", "d", "f"]
    if key in spin_keys:
        return 0
    if key[:1] in orbital_keys:
        return str(orbital_keys.index(key[:1])) + key
    if key.isdecimal():
        return int(key)
    assert key.istitle()  # should be atom
    return 0
