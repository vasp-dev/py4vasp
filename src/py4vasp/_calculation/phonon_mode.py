# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp._calculation import base, structure


class PhononMode(base.Refinery, structure.Mixin):
    """Describes a collective vibration of atoms in a crystal.

    A phonon mode represents a specific way in which atoms in a solid oscillate
    around their equilibrium positions. Each mode is characterized by a frequency
    and a displacement pattern that shows how atoms move relative to each other.
    Low-frequency modes correspond to long-wavelength vibrations, while
    high-frequency modes involve more localized atomic motion."""

    @base.data_access
    def __str__(self):
        phonon_frequencies = "\n".join(
            self._frequency_to_string(index, frequency)
            for index, frequency in enumerate(self.frequencies())
        )
        return f"""\
 Eigenvalues of the dynamical matrix
 -----------------------------------
{phonon_frequencies}
"""

    def _frequency_to_string(self, index, frequency):
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

    @base.data_access
    def to_dict(self):
        """Read structure data and properties of the phonon mode into a dictionary.

        The frequency and eigenvector describe with how atoms move under the influence
        of a particular phonon mode. Structural information is added to understand
        what the displacement correspond to.

        Returns
        -------
        dict
            Structural information, phonon frequencies and eigenvectors.
        """
        return {
            "structure": self._structure.read(),
            "frequencies": self.frequencies(),
            "eigenvectors": self._raw_data.eigenvectors[:],
        }

    @base.data_access
    def frequencies(self):
        "Read the phonon frequencies as a numpy array."
        return self._raw_data.frequencies[:]
