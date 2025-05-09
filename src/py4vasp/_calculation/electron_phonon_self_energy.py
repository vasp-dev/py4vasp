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

    @base.data_access
    def __len__(self):
        return len(self._raw_data.valid_indices)

    def _read_slice_of_data(self, name):
        slice_of_data = getattr(self._raw_data, name)[self._steps]
        return [data[:] for data in slice_of_data]

    @base.data_access
    def get_fan(self, arg):
        iband, ikpt, isp, *more = arg
        st = SparseTensor(
            self._read_slice_of_data("band_kpoint_spin_index"),
            0,
            self._read_slice_of_data("fan"),
        )
        return st[arg]


class SparseTensor:
    def __init__(self, band_kpoint_spin_index, band_start, tensor):
        self.band_kpoint_spin_index = band_kpoint_spin_index
        self.band_start = band_start
        self.tensor = tensor

    def _get_band_kpoint_spin_index(self, iband, ikpt, isp):
        return self.band_kpoint_spin_index[iband - self.band_start][ikpt, isp]

    def __getitem__(self, arg):
        iband, ikpt, isp, *more = arg
        ibks = self._get_band_kpoint_spin_index(iband, ikpt, isp)
        return self.tensor[ibks]
