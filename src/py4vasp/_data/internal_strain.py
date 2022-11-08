# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._data import base, structure


class InternalStrain(base.Refinery, structure.Mixin):
    """The internal strain

    You can use this class to extract the internal strain of a linear
    response calculation.
    """

    @base.data_access
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

    @base.data_access
    def to_dict(self):
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
