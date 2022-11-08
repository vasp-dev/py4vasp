# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._control import base
from py4vasp.data import Structure


class POSCAR(base.InputFile):
    """The POSCAR file defining the structure used in the VASP calculation.

    Parameters
    ----------
    path : str or Path
        Defines where the POSCAR file is stored. If set to None, the file will be kept
        in memory.
    """

    def plot(self, *args, **kwargs):
        """Generate a 3d representation of the structure in the file.

        Parameters
        ----------
        supercell : int or np.ndarray
            If present the structure is replicated the specified number of times
            along each direction.

        Returns
        -------
        Viewer3d
            Visualize the structure as a 3d figure.
        """
        return Structure.from_POSCAR(self).plot(*args, **kwargs)
