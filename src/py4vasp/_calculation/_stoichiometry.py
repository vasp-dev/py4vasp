# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools

import numpy as np

from py4vasp import exception, raw
from py4vasp._calculation import base
from py4vasp._calculation.selection import Selection
from py4vasp._util import check, convert, database, documentation, import_, select

mdtraj = import_.optional("mdtraj")
pd = import_.optional("pandas")

_subscript = "_"


ion_types_documentation = """\
ion_types : Sequence
    Overwrite the ion types present in the raw data."""


class Stoichiometry(base.Refinery):
    """The stoichiometry of the crystal describes how many ions of each type exist in a crystal."""

    @classmethod
    def from_ase(cls, structure):
        """Generate a stoichiometry from the given ase Atoms object."""
        return cls.from_data(raw_stoichiometry_from_ase(structure))

    @base.data_access
    def __str__(self):
        return self.to_string()

    @base.data_access
    @documentation.format(ion_types=ion_types_documentation)
    def to_string(self, ion_types=None):
        """Convert the stoichiometry into a string.

        This method is equivalent to calling str() on the stoichiometry except that it
        allows to overwrite the ion types.

        Parameters
        ----------
        {ion_types}

        Returns
        -------
        str
            String representation of the stoichiometry.
        """
        number_suffix = lambda number: str(number) if number > 1 else ""
        dummy_type = lambda index: f"({chr(ord('A') + index)})"
        return self._create_repr(number_suffix, dummy_type, ion_types)

    @base.data_access
    def _repr_html_(self):
        number_suffix = lambda number: f"<sub>{number}</sub>" if number > 1 else ""
        dummy_type = lambda index: f"<em>{chr(ord('A') + index)}</em>"
        return self._create_repr(number_suffix, dummy_type)

    @base.data_access
    @documentation.format(ion_types=ion_types_documentation)
    def to_dict(self, ion_types=None):
        """Read the stoichiometry and convert it to a dictionary.

        Parameters
        ----------
        {ion_types}

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
        return {**self._default_selection(), **self._specific_selection(ion_types)}

    @base.data_access
    def _to_database(self, *args, **kwargs):
        return {
            "stoichiometry": {
                "ion_types": list(self.ion_types()),
                "number_ion_types": list(self._raw_data.number_ion_types),
                "number_ion_types_primitive": None,  # TODO implement
                "formula": None,  # TODO implement SiO2 check ASE for example conventions for order
                "compound": None,  # TODO implement Si-O
            }
        }

    @base.data_access
    @documentation.format(ion_types=ion_types_documentation)
    def to_frame(self, ion_types=None):
        """Convert the stoichiometry to a DataFrame

        Parameters
        ----------
        {ion_types}

        Returns
        -------
        pd.DataFrame
            The dataframe matches atom label and element type.
        """
        return pd.DataFrame(
            {"name": self.names(ion_types), "element": self.elements(ion_types)}
        )

    @base.data_access
    @documentation.format(ion_types=ion_types_documentation)
    def to_mdtraj(self, ion_types=None):
        """Convert the stoichiometry to a mdtraj.Topology.

        Parameters
        ----------
        {ion_types}

        Returns
        -------
        mdtraj.Topology
            Converts the stoichiometry into an object that can be used for mdtraj.
        """
        df = self.to_frame(ion_types)
        df["serial"] = None
        df["resSeq"] = 0
        df["resName"] = "crystal"
        df["chainID"] = 0
        df["formal_charge"] = 0
        return mdtraj.Topology.from_dataframe(df)

    @base.data_access
    @documentation.format(ion_types=ion_types_documentation)
    def to_POSCAR(self, format_newline="", ion_types=None):
        """Generate the stoichiometry lines for the POSCAR file.

        Parameters
        ----------
        format_newline : str
            If you want to display the POSCAR file in a particular way, you can
            use an additional string to add formatting.
        {ion_types}

        Returns
        -------
        str
            A string used to describe the atoms in the system in the POSCAR file
            augmented by the additional formatting string, if given.
        """
        error_message = "The formatting information must be a string."
        check.raise_error_if_not_string(format_newline, error_message)
        number_ion_types = " ".join(str(x) for x in self._raw_data.number_ion_types)
        if ion_types is None and check.is_none(self._raw_data.ion_types):
            return number_ion_types
        else:
            ion_types = " ".join(self._ion_types(ion_types))
            return ion_types + format_newline + "\n" + number_ion_types

    @base.data_access
    @documentation.format(ion_types=ion_types_documentation)
    def names(self, ion_types=None):
        """Extract the labels of all atoms.

        Parameters
        ----------
        {ion_types}

        Returns
        -------
        list
            List of unique string labeling each ion.
        """
        atom_dict = self.to_dict(ion_types)
        return [val.label for val in atom_dict.values() if _subscript in val.label]

    @base.data_access
    @documentation.format(ion_types=ion_types_documentation)
    def elements(self, ion_types=None):
        """Extract the element of all atoms.

        Parameters
        ----------
        {ion_types}

        Returns
        -------
        list
            List of strings specifying the element of each ion.
        """
        repeated_types = (itertools.repeat(*x) for x in self._type_numbers(ion_types))
        return list(itertools.chain.from_iterable(repeated_types))

    @base.data_access
    @documentation.format(ion_types=ion_types_documentation)
    def ion_types(self, ion_types=None):
        """Return the type of all ions in the system as string.

        Parameters
        ----------
        {ion_types}

        Returns
        -------
        list
            List of unique elements.
        """
        return list(dict.fromkeys(self._ion_types(ion_types)))

    @base.data_access
    def number_atoms(self):
        "Return the number of atoms in the system."
        return np.sum(self._raw_data.number_ion_types)

    def _create_repr(self, number_suffix, dummy_type, ion_types=None):
        ion_string = lambda ion, number: f"{ion}{number_suffix(number)}"
        if ion_types is None and check.is_none(self._raw_data.ion_types):
            number_ion_types = range(len(self._raw_data.number_ion_types))
            ion_types = [dummy_type(i) for i in number_ion_types]
        elif ion_types is None:
            ion_types = self._raw_data.ion_types
        total_type_numbers = self._total_type_numbers(ion_types)
        return "".join(ion_string(*item) for item in total_type_numbers.items())

    def _total_type_numbers(self, ion_types):
        result = {}
        for ion_type, number in self._type_numbers(ion_types):
            result.setdefault(ion_type, 0)
            result[ion_type] += number
        return result

    def _default_selection(self):
        num_atoms = self.number_atoms()
        return {select.all: Selection(indices=slice(0, num_atoms))}

    def _specific_selection(self, ion_types):
        result = {}
        for i, element in enumerate(self.elements(ion_types)):
            result.setdefault(element, Selection(indices=[], label=element))
            result[element].indices.append(i)
            # create labels like Si_1, Si_2, Si_3 (starting at 1)
            label = f"{element}{_subscript}{len(result[element].indices)}"
            result[str(i + 1)] = Selection(indices=[i], label=label)
        return _merge_to_slice_if_possible(result)

    def _type_numbers(self, ion_types):
        return zip(self._ion_types(ion_types), self._raw_data.number_ion_types)

    def _ion_types(self, ion_types):
        ion_types = self._raw_data.ion_types if ion_types is None else ion_types
        if check.is_none(ion_types):
            message = "If the ion types are not defined, you must pass them as argument to the function."
            raise exception.IncorrectUsage(message)
        clean_string = lambda ion_type: convert.text_to_string(ion_type).strip()
        return (clean_string(ion_type) for ion_type in ion_types)


def raw_stoichiometry_from_ase(structure):
    """Convert the given ase Atoms object to a raw.Stoichiometry."""
    number_ion_types = []
    ion_types = []
    for element in structure.symbols:
        if ion_types and ion_types[-1] == element:
            number_ion_types[-1] += 1
        else:
            ion_types.append(element)
            number_ion_types.append(1)
    return raw.Stoichiometry(number_ion_types, ion_types)


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
