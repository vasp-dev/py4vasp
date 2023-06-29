# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp._data import base
from py4vasp._util import convert


class Bandgap(base.Refinery):
    """Extract information about the band extrema during the relaxation or MD simulation.

    Contains utility functions to access the fundamental and optical bandgap as well as
    the k-point coordinates at which these are found.
    """

    def to_dict(self):
        return {
            "fundamental": self._fundamental(),
            "kpoint_VBM": self._kpoint("VBM"),
            "kpoint_CBM": self._kpoint("CBM"),
            "optical": self._optical(),
            "kpoint_optical": self._kpoint("optical"),
            "fermi_energy": self._get("Fermi energy"),
        }

    def _fundamental(self):
        return self._get("conduction band minimum") - self._get("valence band maximum")

    def _optical(self):
        return self._get("optical gap top") - self._get("optical gap bottom")

    def _kpoint(self, label):
        kpoint = [
            self._get(f"kx ({label})"),
            self._get(f"ky ({label})"),
            self._get(f"kz ({label})"),
        ]
        return np.array(kpoint)

    def _get(self, desired_label):
        return next(
            self._raw_data.values[-1, index]
            for index, label in enumerate(self._raw_data.labels[:])
            if convert.text_to_string(label) == desired_label
        )
