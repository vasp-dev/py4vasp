# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools

import mdtraj
import numpy as np
import pandas as pd

from py4vasp import raw
from py4vasp._data import base
from py4vasp._data.selection import Selection
from py4vasp._util import check, convert, select

_subscript = "_"


class Topology(base.Refinery):
    """This class accesses the topology of the crystal.

    At the current stage this only provides access to the name of the atoms in
    the unit cell, but one could extend it to identify logical units like the
    octahedra in perovskites.
    """

    @classmethod
    def from_ase(cls, structure):
        """Generate a Topology from the given ase Atoms object."""
        return cls.from_data(raw_topology_from_ase(structure))

    @base.data_access
    def __str__(self):
        number_suffix = lambda number: str(number) if number > 1 else ""
        return self._create_repr(number_suffix)

    @base.data_access
    def _repr_html_(self):
        number_suffix = lambda number: f"<sub>{number}</sub>" if number > 1 else ""
        return self._create_repr(number_suffix)

    @base.data_access
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

    @base.data_access
    def to_frame(self):
        """Convert the topology to a DataFrame

        Returns
        -------
        pd.DataFrame
            The dataframe matches atom label and element type.
        """
        return pd.DataFrame({"name": self.names(), "element": self.elements()})

    @base.data_access
    def to_mdtraj(self):
        """Convert the topology to a mdtraj.Topology."""
        df = self.to_frame()
        df["serial"] = None
        df["resSeq"] = 0
        df["resName"] = "crystal"
        df["chainID"] = 0
        return mdtraj.Topology.from_dataframe(df)

    @base.data_access
    def to_POSCAR(self, format_newline=""):
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
        check.raise_error_if_not_string(format_newline, error_message)
        ion_types = " ".join(self._ion_types)
        number_ion_types = " ".join(str(x) for x in self._raw_data.number_ion_types)
        return ion_types + format_newline + "\n" + number_ion_types

    @base.data_access
    def names(self):
        """Extract the labels of all atoms."""
        atom_dict = self.to_dict()
        return [val.label for val in atom_dict.values() if _subscript in val.label]

    @base.data_access
    def elements(self):
        """Extract the element of all atoms."""
        repeated_types = (itertools.repeat(*x) for x in self._type_numbers())
        return list(itertools.chain.from_iterable(repeated_types))

    @base.data_access
    def ion_types(self):
        "Return the type of all ions in the system as string."
        return list(dict.fromkeys(self._ion_types))

    @base.data_access
    def number_atoms(self):
        "Return the number of atoms in the system."
        return np.sum(self._raw_data.number_ion_types)

    def _create_repr(self, number_suffix):
        ion_string = lambda ion, number: f"{ion}{number_suffix(number)}"
        return "".join(ion_string(*item) for item in self._total_type_numbers().items())

    def _total_type_numbers(self):
        result = {}
        for ion_type, number in self._type_numbers():
            result.setdefault(ion_type, 0)
            result[ion_type] += number
        return result

    def _default_selection(self):
        num_atoms = self.number_atoms()
        return {select.all: Selection(indices=slice(0, num_atoms))}

    def _specific_selection(self):
        result = {}
        for i, element in enumerate(self.elements()):
            result.setdefault(element, Selection(indices=[], label=element))
            result[element].indices.append(i)
            # create labels like Si_1, Si_2, Si_3 (starting at 1)
            label = f"{element}{_subscript}{len(result[element].indices)}"
            result[str(i + 1)] = Selection(indices=[i], label=label)
        return _merge_to_slice_if_possible(result)

    def _type_numbers(self):
        return zip(self._ion_types, self._raw_data.number_ion_types)

    @property
    def _ion_types(self):
        clean_string = lambda ion_type: convert.text_to_string(ion_type).strip()
        return (clean_string(ion_type) for ion_type in self._raw_data.ion_types)


def raw_topology_from_ase(structure):
    """Convert the given ase Atoms object to a raw.Topology."""
    number_ion_types = []
    ion_types = []
    for element in structure.symbols:
        if ion_types and ion_types[-1] == element:
            number_ion_types[-1] += 1
        else:
            ion_types.append(element)
            number_ion_types.append(1)
    return raw.Topology(number_ion_types, ion_types)


def _merge_to_slice_if_possible(selections):
    for selection in selections.values():
        if _is_slice(selection.indices):
            selection.indices = _to_slice(selection.indices)
    return selections


def _is_slice(indices):
    assert sorted(indices) == indices
    return len(indices) == indices[-1] - indices[0] + 1


def _to_slice(indices):
    return slice(indices[0], indices[-1] + 1)
