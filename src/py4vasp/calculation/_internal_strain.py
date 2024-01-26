# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp.calculation import _base, _structure


class InternalStrain(_base.Refinery, _structure.Mixin):
    """The internal strain is the derivative of energy with respect to displacement and strain.

    The internal strain tensor characterizes the deformation within a material at
    a microscopic level. It is a symmetric 3 x 3 matrix per displacement and
    describes the coupling between the displacement of atoms and the strain on
    the system. Specifically, it reveals how atoms would move under strain or which
    stress occurs when the atoms are displaced. VASP computes the internal strain
    with linear response and this class provides access to the resulting data.
    """

    @_base.data_access
    def __str__(self):
        result = """
Internal strain tensor (eV/Å):
 ion  displ     X           Y           Z          XY          YZ          ZX
---------------------------------------------------------------------------------
"""
        for ion, tensor in enumerate(self._raw_data.internal_strain):
            ion_string = f"{ion + 1:4d}"
            for displacement, matrix in zip("xyz", tensor):
                result += _add_matrix_string(ion_string, displacement, matrix)
                ion_string = "    "
        return result.strip()

    @_base.data_access
    def to_dict(self):
        """Read the internal strain to a dictionary.

        Returns
        -------
        dict
            The dictionary contains the structure of the system. As well as the internal
            strain tensor for all ions. The internal strain is the derivative of the
            energy with respect to ionic position and strain of the cell.
        """
        return {
            "structure": self._structure.read(),
            "internal_strain": self._raw_data.internal_strain[:],
        }


def _add_matrix_string(ion_string, displacement, matrix):
    x, y, z = range(3)
    symmetrized_matrix = (
        matrix[x, x],
        matrix[y, y],
        matrix[z, z],
        0.5 * (matrix[x, y] + matrix[y, x]),
        0.5 * (matrix[y, z] + matrix[z, y]),
        0.5 * (matrix[z, x] + matrix[x, z]),
    )
    matrix_string = " ".join(f"{x:11.5f}" for x in symmetrized_matrix)
    return f"{ion_string}    {displacement} {matrix_string}" + "\n"
