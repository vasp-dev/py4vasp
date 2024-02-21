# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import calculation
from py4vasp._control import base
from py4vasp._third_party import view


class POSCAR(base.InputFile, view.Mixin):
    """The POSCAR file defining the structure used in the VASP calculation.

    Parameters
    ----------
    path : str or Path
        Defines where the POSCAR file is stored. If set to None, the file will be kept
        in memory.
    """

    def to_view(self, supercell=None):
        """Generate a 3d representation of the structure in the file.

        Parameters
        ----------
        supercell : int or np.ndarray
            If present the structure is replicated the specified number of times
            along each direction.

        Returns
        -------
        View
            Visualize the structure as a 3d figure.
        """
        return calculation.structure.from_POSCAR(self).plot(supercell)
