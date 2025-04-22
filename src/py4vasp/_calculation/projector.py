# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import re
import warnings

from py4vasp import exception
from py4vasp._calculation import _stoichiometry, base
from py4vasp._util import convert, documentation, index, select

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

    It is possible to add or subtract different components, e.g., a selection of
    "Ti(d) - O(p)" would project onto the d orbitals of Ti and the p orbitals of O and
    then compute the difference of these two selections.

    If you are unsure about the specific projections that are available, you can use

    >>> calculation.projector.selections()
    {'atom': [...], 'orbital': [...], 'spin': [...]}

    to get a list of all available ones.
"""


class Projector(base.Refinery):
    """The projectors used for atom and orbital resolved quantities.

    This is a utility class that facilitates projecting quantities such as the
    electronic band structure and the DOS on atoms and orbitals. As a user, you can
    investigate the available projections with the :meth:`to_dict` or :meth:`selections`
    methods. The former is useful for scripts, when you need to know which array
    index corresponds to which orbital or atom. The latter describes the available
    selections that you can use in the methods that project on orbitals or atoms.
    """

    _missing_data_message = "No projectors found, please verify the LORBIT tag is set."

    @base.data_access
    def __str__(self):
        if self._raw_data.orbital_types.is_none():
            return "no projectors"
        return f"""projectors:
    atoms: {", ".join(self._stoichiometry().ion_types())}
    orbitals: {", ".join(self._orbital_types())}"""

    def _stoichiometry(self):
        return _stoichiometry.Stoichiometry.from_data(self._raw_data.stoichiometry)

    def _orbital_types(self):
        clean_string = lambda orbital: convert.text_to_string(orbital).strip()
        for orbital in self._raw_data.orbital_types:
            orbital = clean_string(orbital)
            if orbital == "x2-y2":
                yield "dx2y2"
            else:
                yield orbital

    @base.data_access
    def to_dict(self):
        """Return a map from labels to indices in the arrays produced by VASP.

        Returns
        -------
        dict
            A dictionary containing three dictionaries for spin, atom, and orbitals.
            Each of those describes which indices VASP uses to store certain elements
            for projected quantities. If VASP was run without setting :tag:`LORBIT`
            this will return an empty dictionary.

        Examples
        --------

        For nonpolarized SrTiO3 with :tag:`LORBIT = 10, this would work like this

        >>> calculation.projector.to_dict()
        {
            "atom": {
                "Sr": slice(0, 1),
                "Ti": slice(1, 2),
                "O": slice(2, 5),
                "1": slice(0, 1),
                "2": slice(1, 2),
                "3": slice(2, 3),
                "4": slice(3, 4),
                "5": slice(4, 5),
            },
            "orbital": {
                "s": slice(0, 1),
                "p": slice(1, 2),
                "d": slice(2, 3),
                "f": slice(3, 4),
            },
            "spin": {
                "total": slice(0, 1),
            }
        }
        """
        if self._raw_data.orbital_types.is_none():
            return {}
        atom_dict = self._init_atom_dict()
        orbital_dict = self._init_orbital_dict()
        spin_dict = self._init_spin_dict()
        return {"atom": atom_dict, "orbital": orbital_dict, "spin": spin_dict}

    def _init_atom_dict(self):
        return {
            key: value.indices
            for key, value in self._stoichiometry().read().items()
            if key != select.all
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
        if self._is_nonpolarized:
            return {"total": slice(0, 1)}
        if self.is_collinear:
            return {"total": slice(0, 2), "up": slice(0, 1), "down": slice(1, 2)}
        return {
            "total": slice(0, 1),
            "sigma_x": slice(1, 2),
            "sigma_y": slice(2, 3),
            "sigma_z": slice(3, 4),
            "x": slice(1, 2),
            "y": slice(2, 3),
            "z": slice(3, 4),
            "sigma_1": slice(1, 2),
            "sigma_2": slice(2, 3),
            "sigma_3": slice(3, 4),
        }

    @property
    def _is_nonpolarized(self):
        return self._raw_data.number_spin_projections == 1

    @property
    def is_collinear(self):
        """Returns whether the number of spin projections equals 2."""
        return self._raw_data.number_spin_projections == 2

    @property
    def _is_noncollinear(self):
        return self._raw_data.number_spin_projections == 4

    @base.data_access
    def selections(self):
        """Return a dictionary describing what options are available to specify the
        atom, orbital, and spin."""
        dicts = self.to_dict()
        if len(dicts) == 0:
            return dicts
        return {
            "atom": sorted(dicts["atom"], key=self._sort_key),
            "orbital": sorted(dicts["orbital"], key=self._sort_key),
            "spin": sorted(dicts["spin"], key=self._sort_key),
        }

    def _sort_key(self, key):
        spin_keys = [
            "total",
            "up",
            "down",
            "sigma_x",
            "sigma_y",
            "sigma_z",
            "x",
            "y",
            "z",
            "sigma_1",
            "sigma_2",
            "sigma_3",
        ]
        orbital_keys = ["s", "p", "d", "f"]
        if key in spin_keys:
            return 0
        if key[:1] in orbital_keys:
            return str(orbital_keys.index(key[:1])) + key
        if key.isdecimal():
            return int(key)
        assert key.istitle()  # should be atom
        return 0

    @base.data_access
    @documentation.format(selection_doc=selection_doc)
    def project(self, selection, projections):
        """Select a certain subset of the given projections and return them with a
        suitable label.

        Parameters
        ----------
        selection : str
            {selection_doc}
        projections : np.ndarray
            A data array where the first three indices correspond to spin, atom, and
            orbital, respectively. The selection will be parsed and mapped onto the
            corresponding dimension.
        Returns
        -------
        dict
            Each selection receives a label describing its spin, atom, and orbital, where
            default choices are skipped. The value associated to the label contains the
            corresponding subsection of the projections array summed over all remaining
            spin components, atoms, and orbitals.
        """
        if not selection:
            return {}
        self._raise_error_if_orbitals_missing()
        selector = self._make_selector(projections)
        return {
            self._create_label(selector, selection): selector[selection]
            for selection in self._parse_selection(selection)
        }

    def _make_selector(self, projections):
        maps = self.to_dict()
        maps = {1: maps["atom"], 2: maps["orbital"], 0: maps["spin"]}
        try:
            return index.Selector(maps, projections, use_number_labels=True)
        except exception._Py4VaspInternalError as error:
            message = f"""Error reading the projections. Please make sure that the passed
                projections has the right format, i.e., the indices correspond to spin,
                atom, and orbital, respectively."""
            raise exception.IncorrectUsage(message) from error

    def _create_label(self, selector, selection):
        label = selector.label(selection)
        if self._is_noncollinear:
            return label.strip("_total")
        return label

    def _parse_selection(self, selection):
        tree = select.Tree.from_selection(selection)
        for selection in tree.selections():
            if self._is_nonpolarized or self._spin_selected(selection):
                yield selection
            elif self.is_collinear:
                # collinear defaults to two separate projections
                yield selection + ("up",)
                yield selection + ("down",)
            else:
                # noncollinear defaults to total
                yield selection + ("total",)

    def _spin_selected(self, selection):
        return any(
            select.contains(selection, choice) for choice in self._init_spin_dict()
        )

    def _raise_error_if_orbitals_missing(self):
        if self._raw_data.orbital_types.is_none():
            message = "Projectors are not available, rerun VASP setting LORBIT >= 10."
            raise exception.IncorrectUsage(message)
