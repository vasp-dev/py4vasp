from py4vasp.data._base import DataBase, RefinementDescriptor
from py4vasp.data._selection import Selection as _Selection
import py4vasp.exceptions as exception
import py4vasp._util.sanity_check as _check
import py4vasp._util.convert as _convert
import numpy as np
import pandas as pd
import functools
import itertools
import mdtraj


class Topology(DataBase):
    """This class accesses the topology of the crystal.

    At the current stage this only provides access to the name of the atoms in
    the unit cell, but one could extend it to identify logical units like the
    octahedra in perovskites

    Parameters
    ----------
    raw_topology : RawTopology
        A dataclass containing the extracted topology information.
    """

    read = RefinementDescriptor("_to_dict")
    to_dict = RefinementDescriptor("_to_dict")
    to_frame = RefinementDescriptor("_to_frame")
    to_mdtraj = RefinementDescriptor("_to_mdtraj")
    to_poscar = RefinementDescriptor("_to_poscar")
    elements = RefinementDescriptor("_elements")
    ion_types = RefinementDescriptor("_ion_types")
    names = RefinementDescriptor("_names")
    number_atoms = RefinementDescriptor("_number_atoms")
    __str__ = RefinementDescriptor("_to_string")
    _repr_html_ = RefinementDescriptor("_to_html")


def _to_string(raw_topology):
    number_suffix = lambda number: str(number) if number > 1 else ""
    return _create_repr(raw_topology, number_suffix)


def _to_html(raw_topology):
    number_suffix = lambda number: f"<sub>{number}</sub>" if number > 1 else ""
    return _create_repr(raw_topology, number_suffix)


def _create_repr(raw_topology, number_suffix):
    number_strings = (number_suffix(n) for n in raw_topology.number_ion_types)
    return "".join(itertools.chain(*zip(_ion_types(raw_topology), number_strings)))


def _to_dict(raw_topology):
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
    return {**_default_selection(raw_topology), **_specific_selection(raw_topology)}


def _to_frame(raw_topology):
    """Convert the topology to a DataFrame

    Returns
    -------
    pd.DataFrame
        The dataframe matches atom label and element type.
    """
    return pd.DataFrame(
        {"name": _names(raw_topology), "element": _elements(raw_topology)}
    )


def _to_mdtraj(raw_topology):
    """Convert the topology to a mdtraj.Topology."""
    df = _to_frame(raw_topology)
    df["serial"] = None
    df["resSeq"] = 0
    df["resName"] = "crystal"
    df["chainID"] = 0
    return mdtraj.Topology.from_dataframe(df)


def _to_poscar(raw_topology, format_newline=""):
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
    ion_types = " ".join(_ion_types(raw_topology))
    number_ion_types = " ".join(str(x) for x in raw_topology.number_ion_types)
    return ion_types + format_newline + "\n" + number_ion_types


def _names(raw_topology):
    """Extract the labels of all atoms."""
    atom_dict = _to_dict(raw_topology)
    return [val.label for val in atom_dict.values() if _subscript in val.label]


_subscript = "_"


def _elements(raw_topology):
    """Extract the element of all atoms."""
    repeated_types = (itertools.repeat(*x) for x in _type_numbers(raw_topology))
    return list(itertools.chain.from_iterable(repeated_types))


def _ion_types(raw_topology):
    "Return the type of all ions in the system as string."
    clean_string = lambda ion_type: _convert.text_to_string(ion_type).strip()
    return [clean_string(ion_type) for ion_type in raw_topology.ion_types]


def _number_atoms(raw_topology):
    "Return the number of atoms in the system."
    return np.sum(raw_topology.number_ion_types)


def _default_selection(raw_topology):
    num_atoms = _number_atoms(raw_topology)
    return {_Selection.default: _Selection(indices=slice(num_atoms))}


def _specific_selection(raw_topology):
    start = 0
    res = {}
    for ion_type, number in _type_numbers(raw_topology):
        end = start + number
        res[ion_type] = _Selection(indices=slice(start, end), label=ion_type)
        for i in range(start, end):
            # create labels like Si_1, Si_2, Si_3 (starting at 1)
            label = ion_type + _subscript + str(i - start + 1)
            res[str(i + 1)] = _Selection(indices=slice(i, i + 1), label=label)
        start += number
    return res


def _type_numbers(raw_topology):
    return zip(_ion_types(raw_topology), raw_topology.number_ion_types)
