# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)


import numpy as np

from py4vasp._calculation import base, structure


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

    @base.data_access
    def to_numpy(self):
        """Convert NICS to a numpy array.

        Returns
        -------
        np.ndarray
            All components of NICS.
        """
        transposed_nics = np.array(self._raw_data.nics).T
        curr_shape = transposed_nics.shape
        transposed_nics = transposed_nics.reshape((*curr_shape[:-1], 3, 3))
        return transposed_nics
