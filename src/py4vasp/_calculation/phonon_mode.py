# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses

import numpy as np

from py4vasp import raw
from py4vasp._calculation.dispatch import (
    DataSource,
    _dispatch,
    merge_default,
    merge_strings,
    merge_to_database,
    quantity,
)
from py4vasp._calculation.structure import StructureHandler
from py4vasp._raw.models import PhononModeModel
from py4vasp._util import check, convert
from py4vasp._util import masses as mass_table

# ħ² in the units the displacement is expressed in, from ħ = 6.582119569e-16 eV s,
# 1 amu = 1.66053907e-27 kg and 1 Å = 1e-10 m. VASP reports the frequency of a mode as
# the energy ħω, so ħ/ω = ħ²/(ħω) converts it to the square of a normal coordinate.
_HBAR_SQUARED = 0.004180159279779  # eV amu Å²


class PhononModeHandler:
    """Handler for phonon mode data — performs all data access and transformation logic."""

    def __init__(self, raw_phonon_mode: raw.PhononMode):
        self._raw_phonon_mode = raw_phonon_mode

    @classmethod
    def from_data(cls, raw_phonon_mode: raw.PhononMode) -> "PhononModeHandler":
        return cls(raw_phonon_mode)

    def __str__(self) -> str:
        phonon_frequencies = "\n".join(
            self._frequency_to_string(index, frequency)
            for index, frequency in enumerate(self.frequencies())
        )
        return f"""\
 Eigenvalues of the dynamical matrix
 -----------------------------------
{phonon_frequencies}
"""

    def to_dict(self) -> dict:
        return {
            "structure": self._structure().to_dict(),
            "frequencies": self.frequencies(),
            "eigenvectors": self._raw_phonon_mode.eigenvectors[:],
        }

    def to_database(self) -> dict:
        frequencies = (
            self.frequencies()
            if not check.is_none(self._raw_phonon_mode.frequencies)
            else None
        )
        frequencies_real_max = (
            float(np.max(frequencies.real)) if frequencies is not None else None
        )
        frequencies_imag_max = (
            float(np.max(frequencies.imag)) if frequencies is not None else None
        )
        return PhononModeModel(
            frequencies_real_max=frequencies_real_max,
            frequencies_imag_max=frequencies_imag_max,
        )

    def frequencies(self) -> np.ndarray:
        """Read the phonon frequencies as a numpy array."""
        return convert.to_complex(self._raw_phonon_mode.frequencies[:])

    def displacements(self, masses=None) -> np.ndarray:
        """Undo the mass weighting of the eigenvectors.

        Parameters
        ----------
        masses : Sequence[float] | None
            The mass of every atom in atomic mass units. Defaults to the standard
            atomic weight of the element.

        Returns
        -------
        np.ndarray
            How far every atom moves along every direction for every mode. Each pattern
            is scaled to a normal coordinate of 1, i.e. the sum of m u² over all atoms
            and directions is 1.
        """
        masses = self._masses(masses)[:, np.newaxis]
        # VASP reports the eigenvectors of the dynamical matrix, which is the force
        # constant matrix divided by the masses, so the displacement of an atom is the
        # eigenvector divided by the square root of its mass
        displacements = self._eigenvectors() / np.sqrt(masses)
        # VASP normalizes the eigenvectors, so this is a no-op on its output; it makes
        # the normal coordinate exactly 1 for any other input, too
        normal_coordinate = np.sqrt(np.sum(masses * displacements**2, axis=(1, 2)))
        return displacements / normal_coordinate[:, np.newaxis, np.newaxis]

    def displace(self, mode, amplitude, masses=None) -> raw.Structure:
        """Displace the structure along one of the phonon modes.

        Parameters
        ----------
        mode : int
            Index of the mode in the order :py:meth:`frequencies` reports them.
        amplitude : float
            The normal coordinate of the displacement in units of the one at which the
            harmonic energy ½ω²Q² of the mode equals ħω.
        masses : Sequence[float] | None
            The mass of every atom in atomic mass units. Defaults to the standard
            atomic weight of the element.

        Returns
        -------
        raw.Structure
            The equilibrium structure with every atom moved along the mode.
        """
        # ½ω²Q² = ħω is solved by Q = sqrt(2ħ/ω); the sign of the frequency does not
        # enter the energy, so an unstable mode uses the magnitude of its imaginary one
        frequency = np.abs(self.frequencies()[mode])
        normal_coordinate = amplitude * np.sqrt(2 * _HBAR_SQUARED / frequency)
        displacement = normal_coordinate * self.displacements(masses)[mode]
        structure = self._structure()
        inverse_lattice = np.linalg.inv(structure.lattice_vectors())
        positions = structure.positions() + displacement @ inverse_lattice
        return dataclasses.replace(self._raw_phonon_mode.structure, positions=positions)

    def _eigenvectors(self) -> np.ndarray:
        """The eigenvectors shaped (mode, atom, direction).

        VASP uses that shape, the demo data flattens the two trailing axes, and both
        list the three directions of an atom next to each other.
        """
        eigenvectors = np.array(self._raw_phonon_mode.eigenvectors[:])
        return eigenvectors.reshape(len(eigenvectors), -1, 3)

    def _masses(self, masses) -> np.ndarray:
        if masses is not None:
            return np.array(masses)
        return mass_table.of(self._structure()._stoichiometry().elements())

    def _structure(self) -> StructureHandler:
        return StructureHandler.from_data(self._raw_phonon_mode.structure)

    def _frequency_to_string(self, index, frequency) -> str:
        if frequency.real >= frequency.imag:
            label = f"{index + 1:4} f  "
        else:
            label = f"{index + 1:4} f/i"
        frequency = np.abs(frequency)
        freq_meV = f"{frequency * 1000:12.6f} meV"
        eV_to_THz = 241.798934781
        freq_THz = f"{frequency * eV_to_THz:11.6f} THz"
        freq_2PiTHz = f"{2 * np.pi * frequency * eV_to_THz:12.6f} 2PiTHz"
        eV_to_cm1 = 8065.610420
        freq_cm1 = f"{frequency * eV_to_cm1:12.6f} cm-1"
        return f"{label}= {freq_THz} {freq_2PiTHz}{freq_cm1} {freq_meV}"


@quantity("mode", group="phonon")
class PhononMode:
    """Describes a collective vibration of atoms in a crystal.

    A phonon mode represents a specific way in which atoms in a solid oscillate
    around their equilibrium positions. Each mode is characterized by a frequency
    and a displacement pattern that shows how atoms move relative to each other.
    Low-frequency modes correspond to long-wavelength vibrations, while
    high-frequency modes involve more localized atomic motion.

    Examples
    --------
    First, we create some example data so that you can follow along. Please define a
    variable `path` with the path to a directory that does not exist yet. Alternatively,
    use your own data if you have run VASP.

    >>> from py4vasp import demo
    >>> calculation = demo.calculation(path)

    Printing the modes lists every frequency in the units a phonon calculation is
    usually reported in. The first three vanish because they translate the whole
    crystal, which costs no energy

    >>> print(calculation.phonon.mode)
     Eigenvalues of the dynamical matrix
     -----------------------------------
       1 f  =    0.000000 THz     0.000000 2PiTHz    0.000000 cm-1     0.000000 meV
       2 f  =    0.000000 THz...

    A mode marked "f/i" instead of "f" has an imaginary frequency and describes a
    displacement that lowers the energy, so the structure is not at a minimum. The
    example data is stable and has none.
    """

    def __init__(self, source, quantity_name: str = "phonon_mode"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_phonon_mode: raw.PhononMode) -> "PhononMode":
        return cls(source=DataSource(raw_phonon_mode))

    def _handler_factory(self, raw):
        return PhononModeHandler.from_data(raw)

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

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not exist yet.
        Alternatively, use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        >>> calculation.phonon.mode.selections()
        {'phonon_mode': ['default']}
        """
        from py4vasp._raw import definition as raw_module

        return {self._quantity_name: list(raw_module.selections(self._quantity_name))}

    def __str__(self, selection=None) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PhononModeHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self) -> dict:
        """Read structure data and properties of the phonon mode into a dictionary.

        The frequency and eigenvector describe with how atoms move under the influence
        of a particular phonon mode. Structural information is added to understand
        what the displacement correspond to.

        Returns
        -------
        dict
            Structural information, phonon frequencies and eigenvectors.

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not exist yet.
        Alternatively, use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        >>> sorted(calculation.phonon.mode.read())
        ['eigenvectors', 'frequencies', 'structure']

        The eigenvectors give the displacement of every atom along every direction,
        one row per mode

        >>> calculation.phonon.mode.read()["eigenvectors"].shape
        (21, 21)
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            PhononModeHandler.to_dict,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Convenient alias for :py:meth:`read`."""
        return self.read()

    def frequencies(self) -> np.ndarray:
        """Read the phonon frequencies as a numpy array.

        Returns
        -------
        np.ndarray
            The eigenvalues of the dynamical matrix as complex numbers in eV. An
            imaginary part marks an unstable mode.

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not exist yet.
        Alternatively, use your own data if you have run VASP.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        >>> calculation.phonon.mode.frequencies().shape
        (21,)

        The three modes that translate the crystal have zero frequency, and none of
        the others is imaginary

        >>> frequencies = calculation.phonon.mode.frequencies()
        >>> int(np.count_nonzero(frequencies == 0))
        3
        >>> bool(np.all(frequencies.imag == 0))
        True
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            PhononModeHandler.frequencies,
        )

    def _to_database(self) -> dict:
        """Return {quantity[_selection]: handler_result} for database storage."""
        return merge_to_database(
            self._source,
            self._quantity_name,
            PhononModeHandler.from_data,
            PhononModeHandler.to_database,
        )
