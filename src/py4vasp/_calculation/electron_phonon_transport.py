# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp._calculation import base, slice_
from py4vasp._calculation.electron_phonon_chemical_potential import (
    ElectronPhononChemicalPotential,
)
from py4vasp._calculation.electron_phonon_self_energy import ElectronPhononSelfEnergy
from py4vasp._third_party import graph
from py4vasp._util import convert, import_, index, select

pd = import_.optional("pandas")


class ElectronPhononTransportInstance:
    def __init__(self, parent, index):
        self.parent = parent
        self.index = index

    def __str__(self):
        lines = []
        lines.append("electron phonon transport %d" % self.index)
        lines.append("id_name %s"%self.id_name)
        lines.append("id_index %s"%self.id_index)
        # Information about the chemical potential
        mu_tag, mu_val = self.parent.chemical_potential_mu_tag()
        mu_idx = self.id_index[2]-1
        lines.append(f"{mu_tag}: {mu_val[mu_idx]}")
        # Information about the scattering approximation
        scattering_approx = self._get_data("scattering_approximation")
        lines.append(f"scattering_approximation: {scattering_approx}")
        return "\n".join(lines)+"\n"

    def _get_data(self, name):
        return self.parent._get_data(name, self.index)

    def to_dict(self, selection=None):
        names = [
            "temperatures",
            "transport_function",
            "electronic_conductivity",
            "mobility",
            "seebeck",
            "peltier",
            "electronic_thermal_conductivity",
            "scattering_approximation",
        ]
        return {name: self._get_data(name) for name in names}

    def selections(self):
        return self.to_dict().keys()

    def to_graph(self, selection):
        tree = select.Tree.from_selection(selection)
        series = []
        for selection in tree.selections():
            data_ = self._get_data(selection[0]).reshape([-1, 9])
            maps = {
                1: self._init_directions_dict(),
            }
            selector = index.Selector(maps, data_, reduction=np.average)
            y = selector[selection[1:]]
            x = self._get_data("temperatures")
            series.append(graph.Series(x, y, label=selection[0]))
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
        return self.parent.id_name()

    def _make_name_index(self):
        """
        Get corresponding id_index from id_name
        """
        # the -1 is because index conversion between fortran and C
        return dict(zip(self.id_name,self.id_index-1))

class ElectronPhononTransport(base.Refinery):
    "Placeholder for electron phonon transport"

    @base.data_access
    def __str__(self):
        return "electron phonon transport"

    @base.data_access
    def to_dict(self):
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
    def _get_data(self, name, index):
        return getattr(self._raw_data, name)[index][:]

    @base.data_access
    def _get_scalar(self, name, index):
        return getattr(self._raw_data, name)[index][()]

    @base.data_access
    def __len__(self):
        return len(self._raw_data.valid_indices)
    
    @base.data_access
    def valid_delta(self):
        return {self._get_scalar("delta", index) for index in range(len(self))}

    @base.data_access
    def valid_nbands_sum(self):

        return {self._get_scalar("nbands_sum", index) for index in range(len(self))}

    @base.data_access
    def valid_scattering_approximation(self):
        return {
            self._get_data("scattering_approximation", index)
            for index in range(len(self))
        }

    @base.data_access
    def selections(self):
        """Return a dictionary describing what options are available
        to read the electron transport coefficients.
        This is done using the self-energy class."""
        self_energy = ElectronPhononSelfEnergy.from_data(
            self._raw_data.self_energy
        )
        return self_energy.selections()
    
    def _generate_selections(self, selection):
        tree = select.Tree.from_selection(selection)
        for selection in tree.selections():
            yield selection

    @base.data_access
    def chemical_potential_mu_tag(self):
        chemical_potential = ElectronPhononChemicalPotential.from_data(
            self._raw_data.chemical_potential
        )
        return chemical_potential.mu_tag()
    
    @base.data_access
    def select(self, selection):
        """Return a list of ElectronPhononSelfEnergyInstance objects matching the selection.

        Parameters
        ----------
        selection : dict
            Dictionary with keys as selection names (e.g., "nbands_sum", "selfen_approx", "selfen_delta")
            and values as the desired values for those properties.

        Returns
        -------
        list of ElectronPhononSelfEnergyInstance
            Instances that match the selection criteria.
        """
        selected_instances = []
        mu_tag, mu_val = self.chemical_potential_mu_tag()
        for idx in range(len(self)):
            match_all = False
            for sel in self._generate_selections(selection):
                match = True
                sel_dict = dict(zip(sel[::2], sel[1::2]))
                for key, value in sel_dict.items():
                    # Map selection keys to property names
                    if key == "nbands_sum":
                        instance_value = self._get_scalar("nbands_sum", idx)
                        match_this = instance_value == value
                    elif key == "selfen_approx":
                        instance_value = self._get_data("scattering_approximation", idx)
                        match_this = instance_value == value
                    elif key == "selfen_delta":
                        instance_value = self._get_scalar("delta", idx)
                        match_this = abs(instance_value-value)<1e-8
                    elif key == mu_tag:
                        mu_idx = self[idx].id_index[2]-1
                        instance_value = mu_val[mu_idx]
                        match_this = abs(instance_value-float(value))<1e-8
                    else:
                        possible_values = self.selections()
                        raise ValueError(f"Invalid selection {key}. Possible values are {possible_values.keys()}")
                    match = match and match_this
                match_all = match_all or match
            if match_all:
                selected_instances.append(ElectronPhononTransportInstance(self, idx))
        return selected_instances