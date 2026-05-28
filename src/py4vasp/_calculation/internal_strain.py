# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pathlib

from py4vasp import raw
from py4vasp._calculation.dispatch import (
    DataSource,
    FileSource,
    merge_default,
    merge_strings,
    quantity,
)
from py4vasp._calculation.structure import StructureHandler


class InternalStrainHandler:
    """The internal strain is the derivative of energy with respect to displacement and strain."""

    def __init__(self, raw_internal_strain: raw.InternalStrain):
        self._raw_internal_strain = raw_internal_strain

    @classmethod
    def from_data(
        cls, raw_internal_strain: raw.InternalStrain
    ) -> "InternalStrainHandler":
        return cls(raw_internal_strain)

    def __str__(self) -> str:
        result = """
Internal strain tensor (eV/Å):
 ion  displ     X           Y           Z          XY          YZ          ZX
---------------------------------------------------------------------------------
"""
        for ion, tensor in enumerate(self._raw_internal_strain.internal_strain):
            ion_string = f"{ion + 1:4d}"
            for displacement, matrix in zip("xyz", tensor):
                result += _add_matrix_string(ion_string, displacement, matrix)
                ion_string = "    "
        return result.strip()

    def to_dict(self) -> dict:
        """Read the internal strain to a dictionary.

        Returns
        -------
        dict
            The dictionary contains the structure of the system. As well as the internal
            strain tensor for all ions. The internal strain is the derivative of the
            energy with respect to ionic position and strain of the cell.
        """
        structure = StructureHandler.from_data(self._raw_internal_strain.structure)
        return {
            "structure": structure.read(),
            "internal_strain": self._raw_internal_strain.internal_strain[:],
        }


@quantity("internal_strain")
class InternalStrain:
    """The internal strain is the derivative of energy with respect to displacement and strain.

    The internal strain tensor characterizes the deformation within a material at
    a microscopic level. It is a symmetric 3 x 3 matrix per displacement and
    describes the coupling between the displacement of atoms and the strain on
    the system. Specifically, it reveals how atoms would move under strain or which
    stress occurs when the atoms are displaced. VASP computes the internal strain
    with linear response and this class provides access to the resulting data.
    """

    def __init__(self, source, quantity_name: str = "internal_strain"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_internal_strain: raw.InternalStrain) -> "InternalStrain":
        """Create an InternalStrain dispatcher from raw data (convenience for testing)."""
        return cls(source=DataSource(raw_internal_strain))

    @classmethod
    def from_path(cls, path=".") -> "InternalStrain":
        """Create an InternalStrain dispatcher that reads from HDF5 files at *path*."""
        return cls(source=FileSource(path))

    @classmethod
    def from_file(cls, file_name) -> "InternalStrain":
        """Create an InternalStrain dispatcher that reads from a specific HDF5 file."""
        resolved = pathlib.Path(file_name).expanduser().resolve()
        return cls(source=FileSource(resolved.parent, file=file_name))

    @property
    def _path(self):
        return self._source.path

    def __str__(self, selection: str | None = None) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            InternalStrainHandler.from_data,
            InternalStrainHandler.__str__,
        )

    def read(self, selection: str | None = None) -> dict:
        """Read the internal strain to a dictionary.

        Returns
        -------
        dict
            The dictionary contains the structure of the system. As well as the internal
            strain tensor for all ions. The internal strain is the derivative of the
            energy with respect to ionic position and strain of the cell.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            InternalStrainHandler.from_data,
            InternalStrainHandler.to_dict,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Convenient alias for :py:meth:`read`. Please read the documentation there."""
        return self.read(selection=selection)


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
