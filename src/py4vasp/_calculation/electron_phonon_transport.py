# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation import base, slice_
from py4vasp._util import convert, select, import_
pd = import_.optional("pandas")

class ElectronPhononTransportInstance:
    def __init__(self,parent,index):
        self.parent = parent
        self.index = index

    def __str__(self):
        return "electron phonon transport %d"%self.index

    def get_data(self,name):
        return self.parent.read_data(name,self.index)

    def to_dict(self,selection=None):
        return {
            "temperatures": self.get_data("temperatures"),
            "transport_function": self.get_data("transport_function"),
            "electronic_conductivity": self.get_data("electronic_conductivity"),
            "mobility": self.get_data("mobility"),
            "seebeck": self.get_data("seebeck"),
            "peltier": self.get_data("peltier"),
            "electronic_thermal_conductivity": self.get_data("electronic_thermal_conductivity"),
            "scattering_approximation": self.get_data("scattering_approximation"),
        }

    def id_index(self):
        return self.get_data("id_index")

    def id_name(self):
        return self.get_data("id_name")

class ElectronPhononTransport(base.Refinery):
    "Placeholder for electron phonon transport"

    @base.data_access
    def __str__(self):
        return "electron phonon transport"

    @base.data_access
    def to_dict(self,selection=None):
        return {
            "naccumulators": len(self._raw_data.valid_indices),
        }

    @base.data_access
    def id_name(self):
        return self._raw_data.id_name[:]

    @base.data_access
    def __getitem__(self,key):
        #TODO add logic to select instances
        return ElectronPhononTransportInstance(self, key)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @base.data_access
    def read_data(self, name, index):
        return getattr(self._raw_data,name)[index][:]

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

    def _parse_selection(self, selection):
        tree = select.Tree.from_selection(selection)
        return list(tree.selections())
        #for selection in tree.selections():
        #    return selection

