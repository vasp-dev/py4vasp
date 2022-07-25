# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp.data import _base


class PhononDos(_base.Refinery):
    """The phonon density of states (DOS).

    You can use this class to extract the phonon DOS data of a VASP
    calculation. The DOS can also be resolved by direction and atom.
    """

    def to_dict(self):
        return {
            "energies": self._raw_data.energies[:],
            "total": self._raw_data.dos[:],
        }
