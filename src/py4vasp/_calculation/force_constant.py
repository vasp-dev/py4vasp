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
from py4vasp._util import check, convert
from py4vasp._util import masses as mass_table

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
        number_ions = self._structure().number_atoms()
        formatter = _StringFormatter(
            number_ions, self._force_constants, self._free_directions()
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
        result = {
            "structure": self._structure().to_dict(),
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
        system. :py:meth:`eigenvalues` returns the corresponding eigenvalues.

        Returns
        -------
        np.ndarray
            The eigenvectors in the order of ascending eigenvalue. The first index
            selects the eigenvector, the second one the atom, and the third one the
            direction. Atoms frozen by selective dynamics have zero displacement.
        """
        return self._diagonalize()[1]

    def eigenvalues(self) -> np.ndarray:
        """Compute the eigenvalues of the force constant matrix in eV/Å²."""
        return np.linalg.eigvalsh(self._force_constants)

    def frequencies(self, masses=None) -> np.ndarray:
        """Compute the vibrational frequencies ħω in eV from the dynamical matrix."""
        eigenvalues, _ = self._diagonalize_dynamical_matrix(masses)
        # a negative eigenvalue is an unstable mode, which VASP reports as an imaginary
        # frequency
        magnitude = _frequency_magnitude(eigenvalues)
        return np.where(eigenvalues < 0, 1j * magnitude, magnitude + 0j)

    def displacements(self, masses=None) -> np.ndarray:
        """Compute how the atoms move in every normal mode."""
        _, displacements = self._diagonalize_dynamical_matrix(masses)
        return self._unpack(displacements)

    def _diagonalize_dynamical_matrix(self, masses):
        elements = self._structure()._stoichiometry().elements()
        masses = np.repeat(mass_table.resolve(masses, elements), 3)
        inverse_sqrt_mass = 1 / np.sqrt(masses[self._free_directions().flatten()])
        weights = np.outer(inverse_sqrt_mass, inverse_sqrt_mass)
        eigenvalues, eigenvectors = np.linalg.eigh(weights * self._force_constants)
        # the eigenvectors are normalized, so dividing by the square root of the mass
        # gives the displacement u with the sum of m |u|² equal to 1
        displacements = eigenvectors.T * inverse_sqrt_mass
        return eigenvalues, displacements

    def _structure(self):
        return StructureHandler.from_data(self._raw_force_constant.structure)

    def _free_directions(self):
        if check.is_none(self._raw_force_constant.selective_dynamics):
            return np.ones((self._structure().number_atoms(), 3), dtype=np.bool_)
        return self._raw_force_constant.selective_dynamics[:].astype(np.bool_)

    def _diagonalize(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self._force_constants)
        return eigenvalues, self._unpack(eigenvectors.T)

    def _unpack(self, vectors):
        # every vector covers only the free directions; frozen atoms do not move
        free_directions = self._free_directions()
        unpacked = np.zeros((len(vectors), *free_directions.shape))
        unpacked[:, free_directions] = vectors
        return unpacked

    def to_molden(self, masses=None) -> str:
        """Convert the normal modes into molden format, with the frequencies in cm⁻¹."""
        eigenvalues, displacements = self._diagonalize_dynamical_matrix(masses)
        # molden expects wavenumbers and marks an unstable mode by a negative one
        magnitude = _frequency_magnitude(eigenvalues)
        wavenumbers = np.sign(eigenvalues) * magnitude * convert.EV_TO_CM1
        frequencies = "\n".join(f"{x:12.6f}" for x in wavenumbers)
        # a viewer only needs the direction of every mode, so it is normalized to 1
        directions = displacements / np.linalg.norm(displacements, axis=1)[:, None]
        return f"""\
[Molden Format]
[FREQ]
{frequencies}
[FR-COORD]
{self._format_coordinates()}
[FR-NORM-COORD]
{self._format_eigenvectors(self._unpack(directions))}
"""

    def _format_coordinates(self):
        structure = self._structure()
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


def _frequency_magnitude(eigenvalues):
    # an eigenvalue of the dynamical matrix is ω², so ħ² turns it into (ħω)²
    return np.sqrt(convert.HBAR_SQUARED * np.abs(eigenvalues))


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

    The eigenvalues of Φ are not the squares of the vibrational frequencies, because
    the masses of the atoms do not enter. :py:meth:`frequencies` divides Φ by the square
    root of the masses of both atoms and reports the frequencies of the resulting
    dynamical matrix as the energy ħω in eV, the same convention the phonon modes use.
    By default it takes the standard atomic weight of every element, where VASP uses
    the POMASS of the POTCAR, so pass the masses explicitly to match VASP exactly.

    See Also
    --------
    py4vasp._calculation.neighbor_list.NeighborList :
        The distance between the two atoms of each element of Φ. Use it to plot how the
        force constants decay with distance; it is correct for tilted cells, where the
        minimum-image convention ``d - np.rint(d)`` is not.
    py4vasp._calculation.phonon_band.PhononBand :
        The vibrational modes these force constants determine.
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
        system. :py:meth:`eigenvalues` returns the corresponding eigenvalues.

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

    def eigenvalues(self) -> np.ndarray:
        """Compute the eigenvalues of the force constant matrix.

        These are the curvatures of the energy along the directions
        :py:meth:`eigenvectors` returns, in the same order. The masses of the atoms do
        not enter, so they are not the squares of the vibrational frequencies. A
        dynamically stable structure has no negative eigenvalue and, if you displaced
        all atoms, three that vanish because moving the whole crystal does not change
        its energy.

        Returns
        -------
        np.ndarray
            The eigenvalues in eV/Å² in ascending order, one for every direction of
            every atom that was displaced. Atoms frozen by selective dynamics do not
            contribute.

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not exist yet.
        Alternatively, use your own data if you have run VASP.

        >>> import numpy as np
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        The first three eigenvalues belong to the translations of the crystal, so they
        vanish up to numerical noise; the others are positive, because the structure is
        stable.

        >>> eigenvalues = calculation.force_constant.eigenvalues()
        >>> np.allclose(eigenvalues[:3], 0)
        True
        >>> eigenvalues[3:6]
        array([0.813..., 1.089..., 1.639...])
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            ForceConstantHandler.from_data,
            ForceConstantHandler.eigenvalues,
        )

    def frequencies(self, masses=None) -> np.ndarray:
        """Compute the vibrational frequencies of the structure at the zone centre.

        The frequencies follow from the dynamical matrix, i.e. the force constants
        divided by the square root of the masses of both atoms. py4vasp reports them as
        the energy ħω in eV like every other energy, so the numbers compare directly to
        :py:meth:`py4vasp._calculation.phonon_mode.PhononMode.frequencies`. Multiply
        with 241.798934781 to get THz or with 8065.610420 to get cm⁻¹; these are the
        factors VASP and :py:meth:`to_molden` use.

        An unstable mode, i.e. a negative eigenvalue of the dynamical matrix, has an
        imaginary frequency, so the array is complex. py4vasp reports the result as is:
        the three modes that translate the crystal vanish only up to numerical noise and
        may come out as a tiny imaginary number.

        Parameters
        ----------
        masses : Sequence[float] | None
            The mass of every atom in atomic mass units, in the order of the structure.
            Defaults to the standard atomic weight of the element. VASP uses the POMASS
            of the POTCAR instead, so pass those if you changed them or need to match
            the OUTCAR to the last digit.

        Returns
        -------
        np.ndarray
            The complex frequencies ħω in eV in ascending order of the eigenvalue of the
            dynamical matrix, so the unstable modes come first. There is one frequency
            for every direction of every atom that was displaced; atoms frozen by
            selective dynamics do not contribute.

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not exist yet.
        Alternatively, use your own data if you have run VASP.

        >>> import numpy as np
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        The three translations of the crystal have no frequency, and none of the other
        modes is imaginary, because the structure is stable.

        >>> frequencies = calculation.force_constant.frequencies()
        >>> np.abs(frequencies[:3]) < 1e-6
        array([ True,  True,  True])
        >>> frequencies[3:6]
        array([0.0132...+0.j, 0.0148...+0.j, 0.0169...+0.j])

        These are the same frequencies the phonon modes report, converted here to cm⁻¹

        >>> np.round(frequencies[3:6].real * 8065.610420)
        array([107., 120., 137.])

        You can replace the masses, e.g. to study the isotope effect. Doubling every
        mass lowers each frequency by a factor of √2

        >>> masses = 2 * np.array([87.62, 87.62, 47.867, 15.999, 15.999, 15.999, 15.999])
        >>> heavy = calculation.force_constant.frequencies(masses)
        >>> np.allclose(heavy[3:] * np.sqrt(2), frequencies[3:])
        True
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            ForceConstantHandler.from_data,
            ForceConstantHandler.frequencies,
            masses,
        )

    def displacements(self, masses=None) -> np.ndarray:
        """Compute how the atoms move in every normal mode.

        The normal modes are the eigenvectors of the dynamical matrix with the mass
        weighting undone, so they describe the actual displacement of every atom. They
        come in the order of :py:meth:`frequencies`, i.e. the n-th pattern vibrates
        with the n-th frequency. In contrast, :py:meth:`eigenvectors` ignores the
        masses, so its vectors are not the normal modes.

        Parameters
        ----------
        masses : Sequence[float] | None
            The mass of every atom in atomic mass units, in the order of the structure.
            Defaults to the standard atomic weight of the element.

        Returns
        -------
        np.ndarray
            The displacement in Å of every atom along every direction. The first index
            selects the mode, the second one the atom, and the third one the direction.
            Each pattern is scaled to a normal coordinate of 1, i.e. the sum of m |u|²
            over all atoms and directions is 1 amu Å². Atoms frozen by selective
            dynamics do not move. The overall sign of a pattern is arbitrary, as is the
            choice of patterns within a set of degenerate modes.

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not exist yet.
        Alternatively, use your own data if you have run VASP.

        >>> import numpy as np
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        There is one pattern for each of the 21 modes of the seven atoms

        >>> displacements = calculation.force_constant.displacements()
        >>> displacements.shape
        (21, 7, 3)

        Every pattern is normalized with the masses, and apart from the three
        translations the modes leave the centre of mass where it is

        >>> masses = np.array([87.62, 87.62, 47.867, 15.999, 15.999, 15.999, 15.999])
        >>> np.allclose(np.einsum("a,mad->m", masses, displacements**2), 1)
        True
        >>> np.allclose(np.einsum("a,mad->md", masses, displacements[3:]), 0)
        True
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            ForceConstantHandler.from_data,
            ForceConstantHandler.displacements,
            masses,
        )

    def to_molden(self, masses=None) -> str:
        """Convert the normal modes into molden format to animate them in a viewer.

        The file lists the frequency of every mode in cm⁻¹, the positions of the atoms
        in Bohr, and the displacement pattern of every mode. Frequencies and patterns
        are the ones :py:meth:`frequencies` and :py:meth:`displacements` return, i.e.
        the normal modes of the dynamical matrix, so the masses of the atoms enter.
        Following the molden convention, an unstable mode with an imaginary frequency
        is listed as a negative wavenumber, so the three translations may show up as a
        tiny negative number. Each pattern is normalized to a length of 1, because a
        viewer only needs its direction. The molden format has no unit cell, so a
        viewer shows the atoms of one cell as an isolated cluster.

        Parameters
        ----------
        masses : Sequence[float] | None
            The mass of every atom in atomic mass units, in the order of the structure.
            Defaults to the standard atomic weight of the element.

        Returns
        -------
        str
            String describing the structure and normal modes in molden format. Write it
            to a file to open it in a viewer such as molden or Jmol.

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not exist yet.
        Alternatively, use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        The [FREQ] block starts with the three translations of the crystal, which are
        zero up to numerical noise, followed by the optical modes in cm⁻¹

        >>> molden = calculation.force_constant.to_molden()
        >>> lines = molden.splitlines()
        >>> lines[:2]
        ['[Molden Format]', '[FREQ]']
        >>> print("\\n".join(lines[5:8]))
        106.74...
        120.08...
        136.76...

        Write the string to a file to animate the modes

        >>> _ = (path / "modes.molden").write_text(molden)
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            ForceConstantHandler.from_data,
            ForceConstantHandler.to_molden,
            masses,
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
