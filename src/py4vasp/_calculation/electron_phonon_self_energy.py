# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation import base, slice_


class ElectronPhononSelfEnergy(slice_.Mixin, base.Refinery):
    "Placeholder for electron phonon self energy"

    @base.data_access
    def __str__(self):
        return "electron phonon self energy"

    @base.data_access
    def to_dict(self):
        return {
            "eigenvalues": self._raw_data.eigenvalues[:],
            "debye_waller": self._read_slice_of_data("debye_waller"),
            "fan": self._read_slice_of_data("fan"),
        }

    def _read_slice_of_data(self, name):
        slice_of_data = getattr(self._raw_data, name)[self._steps]
        return [data[:] for data in slice_of_data]
