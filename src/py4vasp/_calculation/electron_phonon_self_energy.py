# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation import base, electron_phonon_chemical_potential, slice_


class ElectronPhononSelfEnergyInstance:
    def __init__(self, parent, index):
        self.parent = parent
        self.index = index

    def __str__(self):
        return "electron phonon self energy %d" % self.index

    def get_data(self, name):
        return self.parent.get_data(name, self.index)

    def get_scalar(self, name):
        return self.parent.get_scalar(name, self.index)

    def to_dict(self, selection=None):
        return {
            "eigenvalues": self.parent.eigenvalues(),
            "debye_waller": self.get_data("debye_waller"),
            "fan": self.get_data("fan"),
            "nbands_sum": self.get_scalar("nbands_sum"),
            "delta": self.get_scalar("delta"),
            "scattering_approximation": self.get_data("scattering_approximation"),
        }

    @property
    def id_index(self):
        return self.get_data("id_index")

    @property
    def id_name(self):
        return self.parent.id_name

    @property
    def scattering_approximation(self):
        return self.get_data("scattering_approximation")

    @base.data_access
    def get_fan(self, arg):
        iband, ikpt, isp, *more = arg
        st = SparseTensor(
            self.get_data("band_kpoint_spin_index"),
            0,
            self.get_data("fan"),
        )
        return st[arg]


class ElectronPhononSelfEnergy(base.Refinery, electron_phonon_chemical_potential.Mixin):
    "Placeholder for electron phonon self energy"

    @base.data_access
    def __str__(self):
        return "electron phonon self energy"

    @base.data_access
    def to_dict(self):
        return {"naccumulators": len(self._raw_data.valid_indices)}

    @base.data_access
    def eigenvalues(self):
        return self._raw_data.eigenvalues[:]

    @property
    @base.data_access
    def id_name(self):
        return self._raw_data.id_name[:]

    @property
    @base.data_access
    def id_size(self):
        return self._raw_data.id_size[:]

    @base.data_access
    def __getitem__(self, key):
        # TODO add logic to select instances
        return ElectronPhononSelfEnergyInstance(self, key)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @base.data_access
    def valid_delta(self):
        return {self.get_scalar("delta", index) for index in range(len(self))}

    @base.data_access
    def valid_nbands_sum(self):
        return {self.get_scalar("nbands_sum", index) for index in range(len(self))}

    @base.data_access
    def valid_scattering_approximation(self):
        return {
            self.get_data("scattering_approximation", index)
            for index in range(len(self))
        }

    @base.data_access
    def get_data(self, name, index):
        return getattr(self._raw_data, name)[index][:]

    @base.data_access
    def get_scalar(self, name, index):
        return getattr(self._raw_data, name)[index][()]

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


class Mixin:
    @property
    def _electron_phonon_self_energy(self):
        return ElectronPhononSelfEnergy.from_data(self._raw_data.self_energy)
