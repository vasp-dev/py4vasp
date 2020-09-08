from py4vasp.data import _util
import numpy as np
import pandas as pd
import mdtraj
import functools

_subscript = "_"


@_util.add_wrappers
class Topology:
    """ This class accesses the topology of the crystal.

    At the current stage this only provides access to the name of the atoms in
    the unit cell, but one could extend it to identify logical units like the
    octahedra in perovskites

    Parameters
    ----------
    raw_topology : raw.Topology
        A dataclass containing the extracted topology information.
    """

    def __init__(self, raw_topology):
        self._raw = raw_topology

    @classmethod
    @_util.add_doc(_util.from_file_doc("topology"))
    def from_file(cls, file=None):
        return _util.from_file(cls, file, "topology")

    def to_dict(self):
        """ Read the topology and convert it to a dictionary.

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
        return {
            **self._default_selection(self._raw),
            **self._specific_selection(self._raw),
        }

    def to_frame(self):
        """ Convert the topology to a DataFrame

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
        atom_dict = self.read()
        elements = []
        for key, val in atom_dict.items():
            if key == _util.default_selection:
                continue
            elif isinstance(val.indices, range):
                elements += len(val.indices) * [key]
        return elements

    def _default_selection(self, topology):
        num_atoms = np.sum(topology.number_ion_types)
        return {_util.default_selection: _util.Selection(indices=range(num_atoms))}

    def _specific_selection(self, topology):
        start = 0
        res = {}
        for ion_type, number in zip(topology.ion_types, topology.number_ion_types):
            ion_type = _util.decode_if_possible(ion_type).strip()
            _range = range(start, start + number)
            res[ion_type] = _util.Selection(indices=_range, label=ion_type)
            for i in _range:
                # create labels like Si_1, Si_2, Si_3 (starting at 1)
                label = ion_type + _subscript + str(_range.index(i) + 1)
                res[str(i + 1)] = _util.Selection(indices=(i,), label=label)
            start += number
        return res
