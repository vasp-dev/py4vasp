# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp.data import _base
from py4vasp.data._selection import Selection as _Selection
import py4vasp.exceptions as exception
import py4vasp._util.sanity_check as _check
import py4vasp._util.convert as _convert
import py4vasp._util.selection as _selection
import numpy as np
import pandas as pd
import functools
import itertools
import mdtraj


_subscript = "_"


class Topology(_base.Refinery):
    """This class accesses the topology of the crystal.

    At the current stage this only provides access to the name of the atoms in
    the unit cell, but one could extend it to identify logical units like the
    octahedra in perovskites.
    """

    @_base.data_access
    def __str__(self):
        number_suffix = lambda number: str(number) if number > 1 else ""
        return self._create_repr(number_suffix)

    @_base.data_access
    def _repr_html_(self):
        number_suffix = lambda number: f"<sub>{number}</sub>" if number > 1 else ""
        return self._create_repr(number_suffix)

    @_base.data_access
    def to_dict(self):
        """Read the topology and convert it to a dictionary.

        Returns
        -------
        dict
            A map from particular labels to the corresponding atom indices. For
            every atom a single label of the form *Element*_*Number* (e.g. Sr_1)
            is constructed. In addition there is a map from atoms of a specific
            element type to all indices of that element and from atom-indices
            strings to atom-indices integers. These access strings are used
            throughout all of the refinement classes.
        """
        return {**self._default_selection(), **self._specific_selection()}

    @_base.data_access
    def to_frame(self):
        """Convert the topology to a DataFrame

        Returns
        -------
        pd.DataFrame
            The dataframe matches atom label and element type.
        """
        return pd.DataFrame({"name": self.names(), "element": self.elements()})

    @_base.data_access
    def to_mdtraj(self):
        """Convert the topology to a mdtraj.Topology."""
        df = self.to_frame()
        df["serial"] = None
        df["resSeq"] = 0
        df["resName"] = "crystal"
        df["chainID"] = 0
        return mdtraj.Topology.from_dataframe(df)

    @_base.data_access
    def to_poscar(self, format_newline=""):
        """Generate the topology lines for the POSCAR file.

        Parameters
        ----------
        format_newline : str
            If you want to display the POSCAR file in a particular way, you can
            use an additional string to add formatting.

        Returns
        -------
        str
            A string used to describe the atoms in the system in the POSCAR file
            augmented by the additional formatting string, if given.
        """
        error_message = "The formatting information must be a string."
        _check.raise_error_if_not_string(format_newline, error_message)
        ion_types = " ".join(self.ion_types())
        number_ion_types = " ".join(str(x) for x in self._raw_data.number_ion_types)
        return ion_types + format_newline + "\n" + number_ion_types

    @_base.data_access
    def names(self):
        """Extract the labels of all atoms."""
        atom_dict = self.to_dict()
        return [val.label for val in atom_dict.values() if _subscript in val.label]

    @_base.data_access
    def elements(self):
        """Extract the element of all atoms."""
        repeated_types = (itertools.repeat(*x) for x in self._type_numbers())
        return list(itertools.chain.from_iterable(repeated_types))

    @_base.data_access
    def ion_types(self):
        "Return the type of all ions in the system as string."
        clean_string = lambda ion_type: _convert.text_to_string(ion_type).strip()
        return [clean_string(ion_type) for ion_type in self._raw_data.ion_types]

    @_base.data_access
    def number_atoms(self):
        "Return the number of atoms in the system."
        return np.sum(self._raw_data.number_ion_types)

    def _create_repr(self, number_suffix):
        number_strings = (number_suffix(n) for n in self._raw_data.number_ion_types)
        return "".join(itertools.chain(*zip(self.ion_types(), number_strings)))

    def _default_selection(self):
        num_atoms = self.number_atoms()
        return {_selection.all: _Selection(indices=slice(num_atoms))}

    def _specific_selection(self):
        start = 0
        res = {}
        for ion_type, number in self._type_numbers():
            end = start + number
            res[ion_type] = _Selection(indices=slice(start, end), label=ion_type)
            for i in range(start, end):
                # create labels like Si_1, Si_2, Si_3 (starting at 1)
                label = ion_type + _subscript + str(i - start + 1)
                res[str(i + 1)] = _Selection(indices=slice(i, i + 1), label=label)
            start += number
        return res

    def _type_numbers(self):
        return zip(self.ion_types(), self._raw_data.number_ion_types)
