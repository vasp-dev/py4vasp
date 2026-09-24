# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import exception, raw
from py4vasp._calculation.dispatch import (
    DataSource,
    _dispatch,
    merge_default,
    merge_strings,
    merge_to_database,
    quantity,
)
from py4vasp._calculation.structure import (
    Structure,
    StructureHandler,
    raw_structure_from_parts,
)
from py4vasp._raw.models import PhononModeModel
from py4vasp._util import check, convert
from py4vasp._util import masses as mass_table
from py4vasp._util import select

# ħ² in the units the displacement is expressed in, from ħ = 6.582119569e-16 eV s,
# 1 amu = 1.66053907e-27 kg and 1 Å = 1e-10 m. VASP reports the frequency of a mode as
# the energy ħω, so ħ/ω = ħ²/(ħω) converts it to the square of a normal coordinate.
_HBAR_SQUARED = 0.004180159279779  # eV amu Å²
# Below this energy a mode carries no meaningful scale: it is the numerical zero of
# a translation, far under any vibration of a crystal (1e-5 eV is 0.08 cm⁻¹).
_MINIMUM_FREQUENCY = 1e-5  # eV


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
        masses = self._masses(masses)
        eigenvectors = self._eigenvectors()
        return np.array(
            [self._undo_mass_weighting(vector, masses) for vector in eigenvectors]
        )

    def displace(self, selection=None, amplitude=1.0, masses=None):
        """Displace the structure along one or several phonon modes.

        Parameters
        ----------
        selection : str | None
            Which modes to displace along, by the number with which :py:meth:`print`
            labels them, i.e. counting from 1. Separate several modes by commas. If you
            do not select any, py4vasp displaces along every mode that has a frequency.
        amplitude : float
            The normal coordinate of the displacement in units of the one at which the
            harmonic energy ½ω²Q² of the mode equals ħω.
        masses : Sequence[float] | None
            The mass of every atom in atomic mass units. Defaults to the standard
            atomic weight of the element.

        Returns
        -------
        raw.Structure | dict
            The equilibrium structure with every atom moved along the mode. If you
            select more than one mode, the structures are returned in a dictionary
            using the selected modes as keys.
        """
        masses = self._masses(masses)
        modes = self._select_modes(selection)
        structures = {
            label: self._displace_single_mode(index, amplitude, masses)
            for label, index in modes
        }
        if selection is not None and len(structures) == 1:
            return next(iter(structures.values()))
        return structures

    def _displace_single_mode(self, index, amplitude, masses) -> raw.Structure:
        # ½ω²Q² = ħω is solved by Q = sqrt(2ħ/ω); the sign of the frequency does not
        # enter the energy, so an unstable mode uses the magnitude of its imaginary one
        frequency = np.abs(self.frequencies()[index])
        normal_coordinate = amplitude * np.sqrt(2 * _HBAR_SQUARED / frequency)
        pattern = self._undo_mass_weighting(self._eigenvectors()[index], masses)
        structure = self._structure()
        lattice_vectors = structure.lattice_vectors()
        inverse_lattice = np.linalg.inv(lattice_vectors)
        positions = (
            structure.positions() + normal_coordinate * pattern @ inverse_lattice
        )
        # the raw structure must not reference the HDF5 file, because py4vasp closes it
        # as soon as the displacement is computed
        return raw_structure_from_parts(
            lattice_vectors, positions, structure._stoichiometry().elements()
        )

    def _undo_mass_weighting(self, eigenvector, masses) -> np.ndarray:
        # VASP reports the eigenvectors of the dynamical matrix, which is the force
        # constant matrix divided by the masses, so the displacement of an atom is the
        # eigenvector divided by the square root of its mass
        masses = masses[:, np.newaxis]
        displacement = eigenvector / np.sqrt(masses)
        # VASP normalizes the eigenvectors, so this is a no-op on its output; it makes
        # the normal coordinate exactly 1 for any other input, too
        normal_coordinate = np.sqrt(np.sum(masses * displacement**2))
        return displacement / normal_coordinate

    def _select_modes(self, selection):
        if selection is None:
            return list(self._modes_with_frequency())
        tree = select.Tree.from_selection(selection)
        return [self._single_mode(sel) for sel in tree.selections()]

    def _modes_with_frequency(self):
        for index, frequency in enumerate(np.abs(self.frequencies())):
            if frequency > _MINIMUM_FREQUENCY:
                yield str(index + 1), index

    def _single_mode(self, selection):
        label = self._label_of_mode(selection)
        index = self._index_of_mode(label)
        self._raise_error_if_mode_has_no_frequency(label, index)
        return label, index

    def _label_of_mode(self, selection) -> str:
        if len(selection) == 1 and isinstance(selection[0], str):
            return selection[0]
        message = (
            f"py4vasp cannot displace the structure along '{select.selections_to_string([list(selection)])}' "
            "because it does not describe a single mode. Ranges like '1:3' and "
            "combinations like '1 + 2' are not implemented; please select the modes "
            "one by one, e.g. '1, 2, 3'."
        )
        raise exception.IncorrectUsage(message)

    def _index_of_mode(self, label) -> int:
        number_modes = len(self.frequencies())
        try:
            number = int(label)
        except ValueError:
            number = 0
        if not 1 <= number <= number_modes:
            message = (
                f"'{label}' is not a phonon mode of this structure. Please select a "
                f"mode by the number print labels it with, from 1 to {number_modes}."
            )
            raise exception.IncorrectUsage(message)
        return number - 1

    def _raise_error_if_mode_has_no_frequency(self, label, index):
        if np.abs(self.frequencies()[index]) > _MINIMUM_FREQUENCY:
            return
        message = (
            f"The phonon mode {label} has a frequency of zero, so moving the atoms "
            "along it does not change the energy of the system and the amplitude "
            "has no scale to refer to. The modes of zero frequency translate the "
            "whole crystal; if you want a mode of small but nonzero frequency "
            "instead, keep in mind that the displacement grows like 1/sqrt(ħω)."
        )
        raise exception.IncorrectUsage(message)

    def _eigenvectors(self) -> np.ndarray:
        """The eigenvectors shaped (mode, atom, direction).

        VASP uses that shape, the demo data flattens the two trailing axes, and both
        list the three directions of an atom next to each other.
        """
        eigenvectors = np.array(self._raw_phonon_mode.eigenvectors[:])
        number_atoms = self._structure().number_atoms()
        return eigenvectors.reshape(len(eigenvectors), number_atoms, 3)

    def _masses(self, masses) -> np.ndarray:
        structure = self._structure()
        if masses is None:
            return mass_table.of(structure._stoichiometry().elements())
        masses = np.atleast_1d(masses).ravel()
        number_atoms = structure.number_atoms()
        if len(masses) != number_atoms:
            message = (
                f"You provided {len(masses)} masses but the structure contains "
                f"{number_atoms} atoms. Please pass one mass per atom in the order in "
                "which the structure lists them."
            )
            raise exception.IncorrectUsage(message)
        if not np.all(masses > 0):
            message = (
                "All masses must be positive numbers because the displacement of an "
                f"atom is its eigenvector divided by the square root of its mass; you "
                f"provided {list(masses)}."
            )
            raise exception.IncorrectUsage(message)
        return masses

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

    Note that the eigenvectors VASP reports are the ones of the dynamical matrix, so
    they are weighted with the square root of the mass of the atom and are *not* the
    pattern in which the atoms move. Use :py:meth:`displace` to obtain a structure
    displaced along a mode; it undoes the weighting for you.
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

        These are the eigenvectors of the dynamical matrix, which is the force constant
        matrix divided by the masses, so every atom enters weighted with the square root
        of its mass. They are therefore not the pattern in which the atoms move: an
        acoustic mode translates the whole crystal, moving every atom equally far, and
        yet its eigenvector is largest for the heaviest atom

        >>> acoustic = calculation.phonon.mode.read()["eigenvectors"][0].reshape(-1, 3)
        >>> int(np.argmax(np.linalg.norm(acoustic, axis=1)))
        0

        Dividing by the square root of the mass recovers the displacement, which is the
        same for every atom of that mode

        >>> mass = np.array([87.62, 87.62, 47.867, 15.999, 15.999, 15.999, 15.999])
        >>> distance = np.linalg.norm(acoustic, axis=1) / np.sqrt(mass)
        >>> bool(np.allclose(distance, distance[0]))
        True

        Rather than doing this yourself, use :py:meth:`displace`, which undoes the
        weighting and returns the displaced structure.
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

    def displace(self, selection=None, amplitude=1.0, masses=None):
        """Displace the atoms of the structure along one or several phonon modes.

        Use this to set up a frozen-phonon calculation: displace the structure, write
        the result with :py:meth:`~py4vasp._calculation.structure.Structure.to_POSCAR`
        and run VASP on it. Do not build the displacement from the eigenvectors
        yourself; they are weighted with the square root of the mass of the atom (see
        :py:meth:`read`) and this method undoes that weighting for you.

        Parameters
        ----------
        selection : str | None
            Which modes to displace along. Select a mode by the number with which
            :py:meth:`print` labels it, so the modes count from 1. Separate several
            modes by commas, e.g. "1, 2". If you do not select any mode, py4vasp
            displaces along every mode that has a frequency; the modes that translate
            the whole crystal are skipped because they have no energy scale.
        amplitude : float
            How far to displace the structure along the mode. The unit is the one in
            which the harmonic energy of the mode is its own ħω, so an amplitude of 1
            excites the mode by that energy and one of 2 by four times as much. A
            negative amplitude moves the atoms to the other side of the equilibrium,
            which is what the double well of an unstable mode requires.
        masses : Sequence[float] | None
            The mass of every atom in atomic mass units. By default py4vasp uses the
            standard atomic weight of the element. Set this to the POMASS of your
            POTCAR if you overwrote it, e.g. to replace hydrogen by deuterium; it must
            be the mass VASP used, because that is the one the eigenvectors are
            weighted with.

        Returns
        -------
        Structure | dict[str, Structure]
            The equilibrium structure with every atom moved along the mode. Selecting
            more than one mode returns the structures in a dictionary with the selected
            modes as keys.

        Examples
        --------
        First, we create some example data so that you can follow along. Please define a
        variable `path` with the path to a directory that does not exist yet.
        Alternatively, use your own data if you have run VASP.

        >>> import numpy as np
        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        Printing the modes tells you which number to select; here we take the fourth
        one, the lowest that does not merely translate the crystal. The result behaves
        like any other structure, so `displaced.to_POSCAR()` writes the input of the
        next calculation

        >>> displaced = calculation.phonon.mode.displace("4", amplitude=0.5)
        >>> displaced.read()["elements"]
        ['Sr', 'Sr', 'Ti', 'O', 'O', 'O', 'O']

        The atoms move away from their equilibrium position by the amount the amplitude
        sets. Check how far that is before you spend time on a calculation

        >>> equilibrium = calculation.structure.cartesian_positions()
        >>> shift = displaced.cartesian_positions() - equilibrium
        >>> round(float(np.max(np.linalg.norm(shift, axis=1))), 3)
        0.029

        The displacement is proportional to the amplitude, so scale it if you want a
        particular distance in Å. Here we ask the atom that moves furthest to move by
        0.05 Å

        >>> scale = 0.05 / np.max(np.linalg.norm(shift, axis=1))
        >>> rescaled = calculation.phonon.mode.displace("4", amplitude=0.5 * scale)
        >>> shift = rescaled.cartesian_positions() - equilibrium
        >>> round(float(np.max(np.linalg.norm(shift, axis=1))), 3)
        0.05

        A frozen-phonon scan needs both sides of the minimum, and selecting several
        modes at once gives you a dictionary of structures

        >>> both_sides = calculation.phonon.mode.displace("4, 5", amplitude=-0.5)
        >>> sorted(both_sides)
        ['4', '5']

        The first three modes of this example translate the whole crystal, which costs
        no energy, so there is no amplitude that corresponds to an energy of ħω and
        py4vasp reports an error if you select one of them.
        """
        result = merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PhononModeHandler.displace,
            # keyword arguments, because the dispatcher only passes the selection on
            # when the user made one and would otherwise shift the positional arguments
            amplitude=amplitude,
            masses=masses,
        )
        return _wrap_structures(result)

    def _to_database(self) -> dict:
        """Return {quantity[_selection]: handler_result} for database storage."""
        return merge_to_database(
            self._source,
            self._quantity_name,
            PhononModeHandler.from_data,
            PhononModeHandler.to_database,
        )


def _wrap_structures(result):
    """Turn the raw structures the handler produced into Structure instances."""
    if isinstance(result, dict):
        return {key: _wrap_structures(value) for key, value in result.items()}
    return Structure.from_data(result)
