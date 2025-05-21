# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
from py4vasp._calculation import base, slice_
from py4vasp._third_party import graph
from py4vasp._calculation.electron_phonon_self_energy import ElectronPhononSelfEnergy
from py4vasp._calculation.electron_phonon_chemical_potential import ElectronPhononChemicalPotential
from py4vasp._util import convert, index, import_, select

pd = import_.optional("pandas")


class ElectronPhononTransportInstance:
    def __init__(self, parent, index):
        self.parent = parent
        self.index = index

    def __str__(self):
        return "electron phonon transport %d" % self.index

    def _get_data(self, name):
        return self.parent.read_data(name, self.index)

    def to_dict(self, selection=None):
        names = [
            "temperatures",
            "transport_function",
            "electronic_conductivity",
            "mobility",
            "seebeck",
            "peltier",
            "electronic_thermal_conductivity",
            "scattering_approximation"
        ] 
        return { name : self._get_data(name) for name in names }

    def selections(self):
        return self.to_dict().keys()

    def to_graph(self,selection):
        tree = select.Tree.from_selection(selection)
        series = []
        for selection in tree.selections():
            data_ = self._get_data(selection[0]).reshape([-1,9])
            maps = {
                1: self._init_directions_dict(),
            }
            selector = index.Selector(maps, data_, reduction=np.average)
            y = selector[selection[1:]]
            x = self._get_data("temperatures")
            series.append( graph.Series(x,y,label=selection[0]) )
        return graph.Graph(series) 

    def _init_directions_dict(self):
        return {
            None: [0, 4, 8],
            "isotropic": [0, 4, 8],
            "xx": 0,
            "yy": 4,
            "zz": 8,
            "xy": [1, 3],
            "xz": [2, 6],
            "yz": [5, 7],
        }

    @property
    def id_index(self):
        return self._get_data("id_index")

    @property
    def id_name(self):
        return self.parent.id_name

class ElectronPhononTransport(base.Refinery):
    "Placeholder for electron phonon transport"

    @base.data_access
    def __str__(self):
        return "electron phonon transport"

    @base.data_access
    def to_dict(self, selection=None):
        return {
            "naccumulators": len(self._raw_data.valid_indices),
        }

    @base.data_access
    def id_name(self):
        return self._raw_data.id_name[:]

    @base.data_access
    def id_size(self):
        return self._raw_data.id_size[:]

    @base.data_access
    def __getitem__(self, key):
        # TODO add logic to select instances
        return ElectronPhononTransportInstance(self, key)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @base.data_access
    def read_data(self, name, index):
        return getattr(self._raw_data, name)[index][:]

    @base.data_access
    def __len__(self):
        return len(self._raw_data.valid_indices)

    @base.data_access
    def selections(self):
        """Return a dictionary describing what options are available
        to read the transport coefficients."""
        id_name = self.id_name
        id_size = self.id_size
        self_energy = ElectronPhononSelfEnergy.from_data(self._raw_data.self_energy)
        selections_dict = self_energy.selections()
        chemical_potential = ElectronPhononChemicalPotential.from_data(self._raw_data.chemical_potential)
        mu_tag,mu_val = chemical_potential.mu_tag()
        selections_dict[mu_tag] = mu_val
        return selections_dict

    @base.data_access
    def select(self, selection):
        parsed_selections = self._parse_selection(selection)
        selected_instances = []
        #for elph_selfen_instance in self:
        #    # loop over selections
        #    for parsed_selection in parsed_selections:
        #        print(parsed_selection)
        #    continue
        #    selected_instances.append(elph_selfen_instance)
        return selected_instances

    def _parse_selection(self, selection):
        tree = select.Tree.from_selection(selection)
        return list(tree.selections())
        # for selection in tree.selections():
        #    return selection
