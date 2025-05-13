# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation import base, slice_


class ElectronPhononSelfEnergyInstance:
    def __init__(self,parent,index):
        self.parent = parent
        self.index = index

    def __str__(self):
        return "electron phonon self energy %d"%self.index

    def get_data(self,name):
        return self.parent.read_data(name,self.index)

    def to_dict(self,selection=None):
        return {
            "eigenvalues": self.parent.eigenvalues(),
            "debye_waller": self.get_data("debye_waller"),
            "fan": self.get_data("fan"),
        }

    def id_index(self):
        return self.get_data("id_index")

    def id_name(self):
        return self.get_data("id_name")

    @base.data_access
    def get_fan(self, arg):
        iband, ikpt, isp, *more = arg
        st = SparseTensor(self.get_data("band_kpoint_spin_index"),
            0,
            self.get_data("fan"),
        )
        return st[arg]

class ElectronPhononSelfEnergy(base.Refinery):
    "Placeholder for electron phonon self energy"

    @base.data_access
    def __str__(self):
        return "electron phonon self energy"

    @base.data_access
    def to_dict(self):
        return {
            "naccumulators": len(self._raw_data.valid_indices)
        }

    @base.data_access
    def eigenvalues(self):
        return self._raw_data.eigenvalues[:]

    @base.data_access
    def id_name(self):
        return self._raw_data.id_name[:]

    @base.data_access
    def __getitem__(self,key):
        #TODO add logic to select instances
        return ElectronPhononSelfEnergyInstance(self, key)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @base.data_access
    def read_data(self, name, index):
        return getattr(self._raw_data,name)[index][:]

    @base.data_access
    def __len__(self):
        return len(self._raw_data.valid_indices)


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
