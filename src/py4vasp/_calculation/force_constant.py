# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import itertools
import pathlib

import numpy as np

from py4vasp import raw
from py4vasp._calculation.dispatch import (
    DataSource,
    FileSource,
    merge_default,
    merge_strings,
    quantity,
)
from py4vasp._calculation.structure import StructureHandler
from py4vasp._util import check

_A_TO_BOHR = 0.529177210544


class ForceConstantHandler:
    """Force constants are the 2nd derivatives of the energy with respect to displacement."""

    def __init__(self, raw_force_constant: raw.ForceConstant):
        self._raw_force_constant = raw_force_constant

    @classmethod
    def from_data(cls, raw_force_constant: raw.ForceConstant) -> "ForceConstantHandler":
        return cls(raw_force_constant)

    def __str__(self) -> str:
        structure = StructureHandler.from_data(self._raw_force_constant.structure)
        number_ions = structure.number_atoms()
        force_constants = self._raw_force_constant.force_constants[:]
        force_constants = 0.5 * (force_constants + force_constants.T)
        if check.is_none(self._raw_force_constant.selective_dynamics):
            selective_dynamics = np.ones((number_ions, 3), dtype=np.bool_)
        else:
            selective_dynamics = self._raw_force_constant.selective_dynamics[:]
        return str(_StringFormatter(number_ions, force_constants, selective_dynamics))

    def to_dict(self) -> dict:
        """Read structure information and force constants into a dictionary.

        Returns
        -------
        dict
            Contains structural information as well as the raw force constant data.
        """
        structure = StructureHandler.from_data(self._raw_force_constant.structure)
        result = {
            "structure": structure.read(),
            "force_constants": self._raw_force_constant.force_constants[:],
        }
        if not check.is_none(self._raw_force_constant.selective_dynamics):
            result["selective_dynamics"] = self._raw_force_constant.selective_dynamics[
                :
            ]
        return result

    def eigenvectors(self):
        """Compute the eigenvectors of the force constant matrix."""
        return self._diagonalize()[1]

    def _diagonalize(self):
        eigenvalues, eigenvectors = np.linalg.eigh(
            -self._raw_force_constant.force_constants
        )
        eigenvectors = eigenvectors.T
        if check.is_none(self._raw_force_constant.selective_dynamics):
            return eigenvalues, eigenvectors.reshape(len(eigenvectors), -1, 3)
        structure = StructureHandler.from_data(self._raw_force_constant.structure)
        number_ions = structure.number_atoms()
        unpacked_eigenvectors = np.zeros((len(eigenvectors), number_ions, 3))
        selective_dynamics = self._raw_force_constant.selective_dynamics[:].astype(
            np.bool_
        )
        unpacked_eigenvectors[:, selective_dynamics] = eigenvectors
        return eigenvalues, unpacked_eigenvectors

    def to_molden(self) -> str:
        """Convert the eigenvectors of the force constant into molden format.

        Keep in mind that the eigenvectors indicate the direction of the forces and do
        not take into account the masses of the atom.

        Returns
        -------
        str
            String describing the structure and eigenvectors in molden format.
        """
        eigenvalues, eigenvectors = self._diagonalize()
        frequencies = "\n".join(f"{x:12.6f}" for x in eigenvalues)
        return f"""\
[Molden Format]
[FREQ]
{frequencies}
[FR-COORD]
{self._format_coordinates()}
[FR-NORM-COORD]
{self._format_eigenvectors(eigenvectors)}
"""

    def _format_coordinates(self):
        structure = StructureHandler.from_data(self._raw_force_constant.structure)
        element_positions = zip(
            structure._stoichiometry().elements(),
            structure.cartesian_positions() / _A_TO_BOHR,
        )
        return "\n".join(
            f"{element:2} {self._format_vector(position)}"
            for element, position in element_positions
        )

    def _format_eigenvectors(self, eigenvectors):
        return "\n".join(
            self._format_eigenvector(index, eigenvector)
            for index, eigenvector in enumerate(eigenvectors)
        )

    def _format_eigenvector(self, index, eigenvector):
        sign = np.sign(eigenvector.flatten()[np.argmax(np.abs(eigenvector))])
        eigenvector_string = "\n".join(
            self._format_vector(sign * vector) for vector in eigenvector
        )
        return f"vibration {index + 1}\n{eigenvector_string}"

    def _format_vector(self, vector):
        replace_nearly_zeros = lambda x: 0 if np.isclose(x, 0, atol=1e-9) else x
        return " ".join(f"{replace_nearly_zeros(x):12.6f}" for x in vector)


@quantity("force_constant")
class ForceConstant:
    """Force constants are the 2nd derivatives of the energy with respect to displacement.

    Force constants quantify the strength of interactions between atoms in a crystal
    lattice. They describe how the potential energy of the system changes with atomic
    displacements. Specifically they are the second derivative of the energy with
    respect to a displacement from their equilibrium positions. Force constants are a
    key component in determining the vibrational modes of a crystal lattice (phonon
    dispersion). Phonon calculations involve the computation of these force constants.
    Keep in mind that they are the second derivative at the equilibrium position so
    a careful relaxation is required to eliminate the first derivative (i.e. forces).
    """

    def __init__(self, source, quantity_name: str = "force_constant"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_force_constant: raw.ForceConstant) -> "ForceConstant":
        """Create a ForceConstant dispatcher from raw data (convenience for testing)."""
        return cls(source=DataSource(raw_force_constant))

    @classmethod
    def from_path(cls, path=".") -> "ForceConstant":
        """Create a ForceConstant dispatcher that reads from HDF5 files at *path*."""
        return cls(source=FileSource(path))

    @classmethod
    def from_file(cls, file_name) -> "ForceConstant":
        """Create a ForceConstant dispatcher that reads from a specific HDF5 file."""
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
            ForceConstantHandler.from_data,
            ForceConstantHandler.__str__,
        )

    def read(self, selection: str | None = None) -> dict:
        """Read structure information and force constants into a dictionary.

        The structural information is added to inform about which atoms are included
        in the array. The force constants array contains the second derivatives with
        respect to atomic displacement for all atoms and directions.

        Returns
        -------
        dict
            Contains structural information as well as the raw force constant data.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            ForceConstantHandler.from_data,
            ForceConstantHandler.to_dict,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Convenient alias for :py:meth:`read`."""
        return self.read(selection=selection)

    def eigenvectors(self, selection: str | None = None):
        """Compute the eigenvectors of the force constant matrix."""
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            ForceConstantHandler.from_data,
            ForceConstantHandler.eigenvectors,
        )

    def to_molden(self, selection: str | None = None) -> str:
        """Convert the eigenvectors of the force constant into molden format.

        Keep in mind that the eigenvectors indicate the direction of the forces and do
        not take into account the masses of the atom.

        Returns
        -------
        str
            String describing the structure and eigenvectors in molden format.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            ForceConstantHandler.from_data,
            ForceConstantHandler.to_molden,
        )


@dataclasses.dataclass
class _StringFormatter:
    number_ions: int
    force_constants: np.ndarray
    selective_dynamics: np.ndarray

    def __post_init__(self):
        self.indices = -np.ones(self.selective_dynamics.shape, dtype=np.int32)
        self.indices[self.selective_dynamics] = np.arange(len(self.force_constants))

    def __str__(self):
        return "\n".join(self.line_generator())

    def line_generator(self):
        yield "Force constants (eV/Å²):"
        yield "atom(i)  atom(j)   xi,xj     xi,yj     xi,zj     yi,xj     yi,yj     yi,zj     zi,xj     zi,yj     zi,zj"
        yield "----------------------------------------------------------------------------------------------------------"
        for ion in range(self.number_ions):
            yield from self._ion_to_string(ion)

    def _ion_to_string(self, ion):
        if not any(self.selective_dynamics[ion]):
            yield f"{ion + 1:6d}   frozen"
            return
        for jon in range(ion, self.number_ions):
            if not any(self.selective_dynamics[jon]):
                continue
            yield self._ion_pair_to_string(ion, jon)

    def _ion_pair_to_string(self, ion, jon):
        return (
            f"{ion + 1:6d}   {jon + 1:6d}  {self._force_constants_to_string(ion, jon)}"
        )

    def _force_constants_to_string(self, ion, jon):
        return " ".join(
            self._force_constant_to_string(self.indices[ion, i], self.indices[jon, j])
            for i, j in itertools.product(range(3), repeat=2)
        )

    def _force_constant_to_string(self, index, jndex):
        if index >= 0 and jndex >= 0:
            return f"{self.force_constants[index, jndex]:9.4f}"
        else:
            return "   frozen"
