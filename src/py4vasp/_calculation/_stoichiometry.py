# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools

import numpy as np

from py4vasp import exception, raw
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    merge_strings,
    quantity,
)
from py4vasp._calculation.selection import Selection
from py4vasp._raw.models import StoichiometryModel
from py4vasp._util import check, convert, database, documentation, import_, select

mdtraj = import_.optional("mdtraj")
pd = import_.optional("pandas")

_subscript = "_"


ion_types_documentation = """\
ion_types : Sequence
    Overwrite the ion types present in the raw data."""


class StoichiometryHandler:
    """Processes stoichiometry data from a single raw.Stoichiometry object."""

    def __init__(self, raw_stoichiometry: raw.Stoichiometry):
        self._raw_stoichiometry = raw_stoichiometry

    @classmethod
    def from_data(cls, raw_stoichiometry: raw.Stoichiometry) -> "StoichiometryHandler":
        return cls(raw_stoichiometry)

    def read(self, ion_types=None) -> dict:
        """Read the stoichiometry into a dictionary of selections."""
        return {**self._default_selection(), **self._specific_selection(ion_types)}

    def to_dict(self, ion_types=None) -> dict:
        return self.read(ion_types=ion_types)

    def to_string(self, ion_types=None) -> str:
        """Convert the stoichiometry into a string."""
        number_suffix = lambda number: str(number) if number > 1 else ""
        dummy_type = lambda index: f"({chr(ord('A') + index)})"
        return self._create_repr(number_suffix, dummy_type, ion_types)

    def __str__(self):
        return self.to_string()

    def to_html(self) -> str:
        """HTML representation of the stoichiometry."""
        number_suffix = lambda number: f"<sub>{number}</sub>" if number > 1 else ""
        dummy_type = lambda index: f"<em>{chr(ord('A') + index)}</em>"
        return self._create_repr(number_suffix, dummy_type)

    def to_frame(self, ion_types=None):
        """Convert the stoichiometry to a DataFrame."""
        return pd.DataFrame(
            {"name": self.names(ion_types), "element": self.elements(ion_types)}
        )

    def to_mdtraj(self, ion_types=None):
        """Convert the stoichiometry to a mdtraj.Topology."""
        df = self.to_frame(ion_types)
        df["serial"] = None
        df["resSeq"] = 0
        df["resName"] = "crystal"
        df["chainID"] = 0
        df["formal_charge"] = 0
        return mdtraj.Topology.from_dataframe(df)

    def to_POSCAR(self, format_newline="", ion_types=None) -> str:
        """Generate the stoichiometry lines for the POSCAR file."""
        error_message = "The formatting information must be a string."
        check.raise_error_if_not_string(format_newline, error_message)
        number_ion_types = " ".join(
            str(x) for x in self._raw_stoichiometry.number_ion_types
        )
        if ion_types is None and check.is_none(self._raw_stoichiometry.ion_types):
            return number_ion_types
        else:
            ion_types_str = " ".join(self._ion_types(ion_types))
            return ion_types_str + format_newline + "\n" + number_ion_types

    def names(self, ion_types=None) -> list:
        """Extract the labels of all atoms."""
        atom_dict = self.read(ion_types)
        return [val.label for val in atom_dict.values() if _subscript in val.label]

    def elements(self, ion_types=None) -> list:
        """Extract the element of all atoms."""
        repeated_types = (itertools.repeat(*x) for x in self._type_numbers(ion_types))
        return list(itertools.chain.from_iterable(repeated_types))

    def ion_types_list(self, ion_types=None) -> list:
        """Return the type of all ions in the system as string."""
        return list(dict.fromkeys(self._ion_types(ion_types)))

    def number_atoms(self) -> int:
        """Return the number of atoms in the system."""
        return int(np.sum(self._raw_stoichiometry.number_ion_types))

    def to_database(self) -> StoichiometryModel:
        """Return database-ready stoichiometry data."""
        ion_types = (
            list(self._ion_types(None))
            if not check.is_none(self._raw_stoichiometry.ion_types)
            else None
        )
        num_ion_types = (
            list(self._raw_stoichiometry.number_ion_types)
            if not check.is_none(self._raw_stoichiometry.number_ion_types)
            else None
        )
        formula, compound, simple_types, simple_numbers, primitive_numbers = (
            database.get_formula_and_compound(ion_types, num_ion_types)
        )
        return StoichiometryModel(
            ion_types=simple_types,
            num_ion_types=simple_numbers,
            num_ion_types_primitive=primitive_numbers,
            formula=formula,
            compound=compound,
        )

    def _create_repr(self, number_suffix, dummy_type, ion_types=None):
        ion_string = lambda ion, number: f"{ion}{number_suffix(number)}"
        if ion_types is None and check.is_none(self._raw_stoichiometry.ion_types):
            number_ion_types = range(len(self._raw_stoichiometry.number_ion_types))
            ion_types = [dummy_type(i) for i in number_ion_types]
        elif ion_types is None:
            ion_types = self._raw_stoichiometry.ion_types
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
            label = f"{element}{_subscript}{len(result[element].indices)}"
            result[str(i + 1)] = Selection(indices=[i], label=label)
        return _merge_to_slice_if_possible(result)

    def _type_numbers(self, ion_types):
        return zip(self._ion_types(ion_types), self._raw_stoichiometry.number_ion_types)

    def _ion_types(self, ion_types):
        ion_types = (
            self._raw_stoichiometry.ion_types if ion_types is None else ion_types
        )
        if check.is_none(ion_types):
            message = "If the ion types are not defined, you must pass them as argument to the function."
            raise exception.IncorrectUsage(message)
        clean_string = lambda ion_type: convert.text_to_string(ion_type).strip()
        return (clean_string(ion_type) for ion_type in ion_types)


@quantity("_stoichiometry")
class Stoichiometry:
    """The stoichiometry of the crystal describes how many ions of each type exist in a crystal."""

    def __init__(self, source, quantity_name="stoichiometry"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_stoichiometry):
        return cls(source=DataSource(raw_stoichiometry))

    @classmethod
    def from_ase(cls, structure):
        """Generate a stoichiometry from the given ase Atoms object."""
        return cls.from_data(raw_stoichiometry_from_ase(structure))

    def _handler_factory(self, raw):
        return StoichiometryHandler.from_data(raw)

    def __str__(self, selection=None):
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            StoichiometryHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")

    def _repr_html_(self):
        return merge_strings(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StoichiometryHandler.to_html,
        )

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
        return merge_strings(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StoichiometryHandler.to_string,
            ion_types,
        )

    def read(self, ion_types=None):
        "Convenient wrapper around to_dict. Check that function for examples and optional arguments."
        return self.to_dict(ion_types=ion_types)

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
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StoichiometryHandler.to_dict,
            ion_types,
        )

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
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StoichiometryHandler.to_frame,
            ion_types,
        )

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
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StoichiometryHandler.to_mdtraj,
            ion_types,
        )

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
        return merge_strings(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StoichiometryHandler.to_POSCAR,
            format_newline,
            ion_types,
        )

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
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StoichiometryHandler.names,
            ion_types,
        )

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
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StoichiometryHandler.elements,
            ion_types,
        )

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
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StoichiometryHandler.ion_types_list,
            ion_types,
        )

    def number_atoms(self):
        "Return the number of atoms in the system."
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StoichiometryHandler.number_atoms,
        )

    # Stoichiometry has no standalone database entry: its data is folded into the
    # structure model (see StructureHandler.to_database), so the dispatcher
    # deliberately exposes no `_to_database`.


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
