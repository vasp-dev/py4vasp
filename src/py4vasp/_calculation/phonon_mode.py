# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from py4vasp._calculation import base, structure


class PhononMode(base.Refinery, structure.Mixin):
    """Describes a collective vibration of atoms in a crystal.

    A phonon mode represents a specific way in which atoms in a solid oscillate
    around their equilibrium positions. Each mode is characterized by a frequency
    and a displacement pattern that shows how atoms move relative to each other.
    Low-frequency modes correspond to long-wavelength vibrations, while
    high-frequency modes involve more localized atomic motion."""

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
