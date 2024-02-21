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

    def to_view(self, supercell=None, *, elements=None):
        """Generate a 3d representation of the structure in the file.

        Parameters
        ----------
        supercell : int or np.ndarray
            If present the structure is replicated the specified number of times
            along each direction.

        elements : list[str]
            Name of the elements in the order they appear in the POSCAR file. If the
            elements are specified in the POSCAR file, this argument is optional and
            if set it will overwrite the choice in the POSCAR file. Old POSCAR files
            do not specify the name of the elements; in that case this argument is
            required.

        Returns
        -------
        View
            Visualize the structure as a 3d figure.
        """
        structure = calculation.structure.from_POSCAR(self, elements=elements)
        return structure.plot(supercell)
