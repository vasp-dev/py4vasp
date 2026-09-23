# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses
import itertools

import numpy as np

from py4vasp import raw
from py4vasp._calculation.dispatch import (
    DataSource,
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
        # VASP stores the derivative of the force ∂F/∂u, which is the negative of the
        # Hessian ∂²E/∂u∂u that py4vasp reports everywhere. Symmetrize here as well, so
        # that all methods of this class work on the same matrix.
        force_constants = -np.array(raw_force_constant.force_constants[:])
        self._force_constants = 0.5 * (force_constants + force_constants.T)

    @classmethod
    def from_data(cls, raw_force_constant: raw.ForceConstant) -> "ForceConstantHandler":
        return cls(raw_force_constant)

    def __str__(self) -> str:
        structure = StructureHandler.from_data(self._raw_force_constant.structure)
        number_ions = structure.number_atoms()
        if check.is_none(self._raw_force_constant.selective_dynamics):
            selective_dynamics = np.ones((number_ions, 3), dtype=np.bool_)
        else:
            selective_dynamics = self._raw_force_constant.selective_dynamics[:]
        formatter = _StringFormatter(
            number_ions, self._force_constants, selective_dynamics
        )
        return str(formatter)

    def to_dict(self) -> dict:
        """Read structure information and force constants into a dictionary.

        The force constants are the second derivatives of the energy in eV/Å², i.e.
        the negative of the array VASP stores.

        Returns
        -------
        dict
            Contains structural information as well as the raw force constant data.
        """
        structure = StructureHandler.from_data(self._raw_force_constant.structure)
        result = {
            "structure": structure.to_dict(),
            "force_constants": self._force_constants,
        }
        if not check.is_none(self._raw_force_constant.selective_dynamics):
            result["selective_dynamics"] = self._raw_force_constant.selective_dynamics[
                :
            ]
        return result

    def eigenvectors(self):
        """Compute the eigenvectors of the force constant matrix.

        The eigenvectors are the ones of the force constants themselves; the masses of
        the atoms are not taken into account, so these are not the normal modes of the
        system. Use :py:meth:`read` and diagonalize the resulting array yourself if you
        need the corresponding eigenvalues.

        Returns
        -------
        np.ndarray
            The eigenvectors in the order of ascending eigenvalue. The first index
            selects the eigenvector, the second one the atom, and the third one the
            direction. Atoms frozen by selective dynamics have zero displacement.
        """
        return self._diagonalize()[1]

    def _diagonalize(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self._force_constants)
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

    py4vasp uses the Hessian Φ = ∂²E/∂u∂u throughout, the convention shared by the
    phonon literature and by codes such as phonopy. VASP stores the derivative of the
    force ∂F/∂u = -Φ, so the array in the HDF5 file has the opposite sign of the one
    py4vasp reports. The force constants are in eV/Å² and are symmetrized, because Φ
    is symmetric by construction.

    If you displaced all atoms of a dynamically stable structure, Φ is positive
    semidefinite: no eigenvalue is negative and three of them vanish, because moving
    the whole system does not change its energy. Use this as a check of your
    calculation, but note that it does not apply if you freeze some atoms with
    selective dynamics, because then the translation of the whole system is not
    contained in the force constants.
    """

    def __init__(self, source, quantity_name: str = "force_constant"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_force_constant: raw.ForceConstant) -> "ForceConstant":
        """Create a ForceConstant dispatcher from raw data (convenience for testing)."""
        return cls(source=DataSource(raw_force_constant))

    def print(self, selection: str | None = None) -> None:
        """Print a string representation of this quantity.

        Parameters
        ----------
        selection : str | None
            Select which source of the quantity is printed. If you select multiple
            sources, py4vasp prints one block per source.
        """
        print(self.__str__(selection))

    def selections(self) -> dict:
        """Returns possible alternatives for this particular quantity VASP can produce.

        The returned dictionary contains a single item with the name of the quantity
        mapping to all possible selections. Each of these selections may be passed to
        the other methods of this quantity to choose which output of VASP is used.

        Returns
        -------
        dict
            The key indicates this quantity and the value lists the possible choices
            for the selection argument of its other methods.
        """
        from py4vasp._raw import definition as raw_module

        return {self._quantity_name: list(raw_module.selections(self._quantity_name))}

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def __str__(self, selection=None) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            ForceConstantHandler.from_data,
            ForceConstantHandler.__str__,
        )

    def read(self) -> dict:
        """Read structure information and force constants into a dictionary.

        The structural information is added to inform about which atoms are included
        in the array. The force constants array contains the second derivatives of the
        energy with respect to atomic displacement for all atoms and directions in
        eV/Å². Note that this is the negative of the array VASP stores, see the class
        documentation for the sign convention.

        Returns
        -------
        dict
            Contains structural information as well as the raw force constant data.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            ForceConstantHandler.from_data,
            ForceConstantHandler.to_dict,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Convenient alias for :py:meth:`read`."""
        return self.read()

    def eigenvectors(self):
        """Compute the eigenvectors of the force constant matrix.

        The eigenvectors are the ones of the force constants themselves; the masses of
        the atoms are not taken into account, so these are not the normal modes of the
        system. Use :py:meth:`read` and diagonalize the resulting array yourself if you
        need the corresponding eigenvalues.

        Returns
        -------
        np.ndarray
            The eigenvectors in the order of ascending eigenvalue. The first index
            selects the eigenvector, the second one the atom, and the third one the
            direction. Atoms frozen by selective dynamics have zero displacement.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            ForceConstantHandler.from_data,
            ForceConstantHandler.eigenvectors,
        )

    def to_molden(self) -> str:
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
            None,
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
