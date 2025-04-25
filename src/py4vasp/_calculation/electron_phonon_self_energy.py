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

    def select(self, selection):
        tree = select.Tree.from_string(selection)
        for selection in tree.selections():
            print(selection)

    def get_fan(self, arg):
        iband, ikpt, isp, *more = arg
        st = SparseTensor(self._raw_data.bks_idx, self._raw_data.fan)
        return st(iband, ikpt, isp)


class SparseTensor:
    def __init__(self, bks_idx, tensor):
        self.bks_idx = bks_idx
        self.tensor = tensor

    def _get_ibks(self, iband, ikpt, isp):
        return self.bks_idx[iband, ikpt, isp]

    def __getitem__(self, arg):
        iband, ikpt, isp, *more = arg
        ibks = self._get_ibks(iband, ikpt, isp)
        return self.tensor[ibks, more]
