from py4vasp.data import _util
import py4vasp.exceptions as exception
import numpy as np
import pandas as pd
import mdtraj
import functools
import itertools

_subscript = "_"


@_util.add_wrappers
class Topology(_util.Data):
    """This class accesses the topology of the crystal.

    At the current stage this only provides access to the name of the atoms in
    the unit cell, but one could extend it to identify logical units like the
    octahedra in perovskites

    Parameters
    ----------
    raw_topology : raw.Topology
        A dataclass containing the extracted topology information.
    """

    def __init__(self, raw_topology):
        super().__init__(raw_topology)

    @classmethod
    @_util.add_doc(_util.from_file_doc("topology"))
    def from_file(cls, file=None):
        return _util.from_file(cls, file, "topology")

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

    def to_frame(self):
        """Convert the topology to a DataFrame

        Returns
        -------
        pd.DataFrame
            The dataframe matches atom label and element type.
        """
        return pd.DataFrame({"name": self.names(), "element": self.elements()})

    def to_mdtraj(self):
        """ Convert the topology to a mdtraj.Topology. """
        df = self.to_frame()
        df["serial"] = None
        df["resSeq"] = 0
        df["resName"] = "crystal"
        df["chainID"] = 0
        return mdtraj.Topology.from_dataframe(df)

    def names(self):
        """ Extract the labels of all atoms. """
        atom_dict = self.read()
        return [val.label for val in atom_dict.values() if _subscript in val.label]

    def elements(self):
        """ Extract the element of all atoms. """
        type_numbers = zip(self._ion_types(), self._raw.number_ion_types)
        repeated_types = (itertools.repeat(*x) for x in type_numbers)
        return list(itertools.chain.from_iterable(repeated_types))

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
        _util.raise_error_if_not_string(format_newline, error_message)
        ion_types = " ".join(self._ion_types())
        number_ion_types = " ".join(str(x) for x in self._raw.number_ion_types)
        return ion_types + format_newline + "\n" + number_ion_types

    def _repr_pretty_(self, p, cycle):
        to_string = lambda number: str(number) if number > 1 else ""
        p.text(self._create_repr(to_string))

    def _repr_html_(self):
        to_string = lambda number: f"<sub>{number}</sub>" if number > 1 else ""
        return self._create_repr(to_string)

    def _create_repr(self, to_string):
        number_strings = (to_string(number) for number in self._raw.number_ion_types)
        return "".join(itertools.chain(*zip(self._ion_types(), number_strings)))

    def _default_selection(self):
        num_atoms = np.sum(self._raw.number_ion_types)
        return {_util.default_selection: _util.Selection(indices=slice(num_atoms))}

    def _specific_selection(self):
        start = 0
        res = {}
        for ion_type, number in zip(self._ion_types(), self._raw.number_ion_types):
            end = start + number
            res[ion_type] = _util.Selection(indices=slice(start, end), label=ion_type)
            for i in range(start, end):
                # create labels like Si_1, Si_2, Si_3 (starting at 1)
                label = ion_type + _subscript + str(i - start + 1)
                res[str(i + 1)] = _util.Selection(indices=slice(i, i + 1), label=label)
            start += number
        return res

    def _ion_types(self):
        clean_string = lambda ion_type: _util.decode_if_possible(ion_type).strip()
        return (clean_string(ion_type) for ion_type in self._raw.ion_types)
