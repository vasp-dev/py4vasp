# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import raw
from py4vasp._calculation.dispatch import (
    _dispatch,
    DataSource,
    merge_to_database,
    merge_default,
    merge_strings,
    quantity,
)
from py4vasp._calculation.structure import StructureHandler
from py4vasp._raw.data_db import PhononMode_DB
from py4vasp._util import check, convert


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
        return PhononMode_DB(
            frequencies_real_max=frequencies_real_max,
            frequencies_imag_max=frequencies_imag_max,
        )

    def frequencies(self) -> np.ndarray:
        """Read the phonon frequencies as a numpy array."""
        return convert.to_complex(self._raw_phonon_mode.frequencies[:])

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
    high-frequency modes involve more localized atomic motion."""

    def __init__(self, source, quantity_name: str = "phonon_mode"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_phonon_mode: raw.PhononMode) -> "PhononMode":
        return cls(source=DataSource(raw_phonon_mode))

    def _handler_factory(self, raw):
        return PhononModeHandler.from_data(raw)

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
        """Read the phonon frequencies as a numpy array."""
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
