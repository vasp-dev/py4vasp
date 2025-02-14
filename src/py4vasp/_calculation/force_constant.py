# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import itertools

import numpy as np

from py4vasp._calculation import base, structure
from py4vasp._util import check


class ForceConstant(base.Refinery, structure.Mixin):
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

    @base.data_access
    def __str__(self):
        result = """
Force constants (eV/Å²):
atom(i)  atom(j)   xi,xj     xi,yj     xi,zj     yi,xj     yi,yj     yi,zj     zi,xj     zi,yj     zi,zj
----------------------------------------------------------------------------------------------------------
""".strip()
        number_ions = self._structure.number_atoms()
        force_constants = self._raw_data.force_constants[:]
        force_constants = 0.5 * (force_constants + force_constants.T)
        if check.is_none(self._raw_data.selective_dynamics):
            selective_dynamics = np.ones((number_ions, 3), dtype=np.bool_)
        else:
            selective_dynamics = self._raw_data.selective_dynamics[:]
        return str(_StringFormatter(number_ions, force_constants, selective_dynamics))

    @base.data_access
    def to_dict(self):
        """Read structure information and force constants into a dictionary.

        The structural information is added to inform about which atoms are included
        in the array. The force constants array contains the second derivatives with
        respect to atomic displacement for all atoms and directions.

        Returns
        -------
        dict
            Contains structural information as well as the raw force constant data.
        """
        return {
            "structure": self._structure.read(),
            "force_constants": self._raw_data.force_constants[:],
            **self._read_selective_dynamics(),
        }

    def _read_selective_dynamics(self):
        if not check.is_none(self._raw_data.selective_dynamics):
            return {"selective_dynamics": self._raw_data.selective_dynamics[:]}
        else:
            return {}

    @base.data_access
    def eigenvectors(self):
        """Compute the eigenvectors of the force constant matrix."""
        return self._diagonalize()[1]

    def _diagonalize(self):
        eigenvalues, eigenvectors = np.linalg.eigh(-self._raw_data.force_constants)
        eigenvectors = eigenvectors.T
        if check.is_none(self._raw_data.selective_dynamics):
            return eigenvalues, eigenvectors.reshape(len(eigenvectors), -1, 3)
        number_ions = self._structure.number_atoms()
        unpacked_eigenvectors = np.zeros((len(eigenvectors), number_ions, 3))
        selective_dynamics = self._raw_data.selective_dynamics[:].astype(np.bool_)
        unpacked_eigenvectors[:, selective_dynamics] = eigenvectors
        return eigenvalues, unpacked_eigenvectors

    @base.data_access
    def to_molden(self):
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
        A_to_bohr = 0.529177210544
        element_positions = zip(
            self._structure._stoichiometry().elements(),
            self._structure.cartesian_positions() / A_to_bohr,
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
        eigenvector_string = "\n".join(
            self._format_vector(vector) for vector in eigenvector
        )
        return f"vibration {index + 1}\n{eigenvector_string}"

    def _format_vector(self, vector):
        return " ".join(f"{x:12.6f}" for x in vector)


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
