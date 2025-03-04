# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)


import numpy as np

from py4vasp._calculation import base, structure
from py4vasp._util import index, select


class Nics(base.Refinery, structure.Mixin):
    """This class accesses information on the nucleus-independent chemical shift (NICS)."""

    @base.data_access
    def to_dict(self):
        """Read nics into a dictionary.

        Parameters
        ----------

        Returns
        -------
        dict
            Contains the structure information as well as the nucleus-independent chemical shift represented on a grid in the unit cell.
        """
        result = {
            "structure": self._structure.read(),
            "nics": self.to_numpy(),
        }
        return result

    def _init_directions_dict(self):
        return {
            "isotropic": [0, 4, 8],
            "xx": 0,
            "xy": 1,
            "xz": 2,
            "yx": 3,
            "yy": 4,
            "yz": 5,
            "zx": 6,
            "zy": 7,
            "zz": 8,
        }

    @base.data_access
    def to_numpy(self, selection=None):
        """Convert NICS to a numpy array.

        Parameters
        ----------
        selection : str or None
            The tensor element(s) to extract.
            Can be None (in which case the whole tensor is returned), isotropic, or one of "xx", "xy", ...

        Returns
        -------
        np.ndarray
            All components of NICS.
        """
        transposed_nics = np.array(self._raw_data.nics).T
        curr_shape = transposed_nics.shape
        if selection is None:
            transposed_nics = transposed_nics.reshape((*curr_shape[:-1], 3, 3))
        else:
            selector = index.Selector(
                {3: self._init_directions_dict()}, transposed_nics, reduction=np.average
            )
            tree = select.Tree.from_selection(selection)
            transposed_nics = np.squeeze(
                [selector[selection] for selection in tree.selections()]
            )
        return transposed_nics
