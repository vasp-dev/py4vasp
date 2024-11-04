# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp.calculation import _base, _slice


class ElectronPhononTransport(_slice.Mixin, _base.Refinery):
    "Placeholder for electron phonon transport"

    @_base.data_access
    def __str__(self):
        return "electron phonon transport"

    @_base.data_access
    def to_dict(self):
        return {
            "transport_function": self._read_slice_of_data("transport_function"),
            "mobility": self._read_slice_of_data("mobility"),
        }

    def _read_slice_of_data(self, name):
        slice_of_data = getattr(self._raw_data, name)[self._steps]
        return [data[:] for data in slice_of_data]
