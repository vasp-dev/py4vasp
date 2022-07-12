# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp.data import _base


class PhononBand(_base.Refinery):
    """The phonon band structure.

    Use this to examine the phonon band structure along a high-symmetry path in the
    Brillouin zone. The `to_dict` function allows to extract the raw data to process
    it further."""

    def to_dict(self):
        return {
            "bands": self._raw_data.dispersion.eigenvalues[:],
            "modes": self._raw_data.eigenvectors[:],
        }
