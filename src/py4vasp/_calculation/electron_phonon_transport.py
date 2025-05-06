# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation import base, slice_


class ElectronPhononTransport(slice_.Mixin, base.Refinery):
    "Placeholder for electron phonon transport"

    @base.data_access
    def __str__(self):
        return "electron phonon transport"

    @base.data_access
    def to_dict(self):
        return {
            "temperatures": self._raw_data.temperatures[self._steps][:],
            "transport_function": self._read_slice_of_data("transport_function"),
            "electronic_conductivity": self._read_slice_of_data("electronic_conductivity"),
            "mobility": self._read_slice_of_data("mobility"),
            "seebeck": self._read_slice_of_data("seebeck"),
            "peltier": self._read_slice_of_data("peltier"),
            "electronic_thermal_conductivity": self._read_slice_of_data("electronic_thermal_conductivity"),
            "scattering_approximation": self._raw_data.scattering_approximation[self._steps],
        }

    @base.data_access
    def __len__(self):
        return len(self._raw_data.valid_indices)

    @base.data_access
    def selections(self):
        """Return a dictionary describing what options are available
        to read the transport coefficients."""
        id_name = self._raw_data.id_name
        id_size = self._raw_data.id_size[:]
        return {
            convert.text_to_string(name).strip(): int(size)
            for name, size in zip(id_name, id_size)
        }

    def _read_slice_of_data(self, name):
        slice_of_data = getattr(self._raw_data, name)[self._steps]
        return [data[:] for data in slice_of_data]
