# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp.calculation import _base, _slice


class ElectronPhononSelfEnergy(_slice.Mixin, _base.Refinery):
    "Placeholder for electron phonon self energy"

    def to_dict(self):
        return {
            "eigenvalues": self._raw_data.eigenvalues[:],
            "debye_waller": self._read_slice_of_data("debye_waller"),
            "fan": self._read_slice_of_data("fan"),
        }

    def _read_slice_of_data(self, name):
        slice_of_data = getattr(self._raw_data, name)[self._steps]
        return [data[:] for data in slice_of_data]
