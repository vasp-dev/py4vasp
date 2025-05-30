# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation.electron_phonon_chemical_potential import (
    ElectronPhononChemicalPotential
)
from py4vasp._calculation.electron_phonon_self_energy import ElectronPhononSelfEnergy
from py4vasp._calculation import base, slice_
from py4vasp._third_party import graph
from py4vasp._util import import_, select


class ElectronPhononBandgapInstance:
    "Placeholder for electron phonon band gap"

    def __init__(self, parent, index):
        self.parent = parent
        self.index = index

    def __str__(self):
        return "electron phonon band gap %d" % self.index

    def _get_data(self, name):
        return self.parent._get_data(name, self.index)

    def _get_scalar(self, name):
        return self.parent._get_scalar(name, self.index)

    def to_graph(self, selection):
        tree = select.Tree.from_selection(selection)
        series = []
        for selection in tree.selections():
            y = self._get_data(selection[0])
            x = self._get_data("temperatures")
            series.append(graph.Series(x, y, label=selection[0]))
        return graph.Graph(series, ylabel="energy (eV)", xlabel="Temperature (K)")

    def to_dict(self):
        _dict = {
            "nbands_sum": self._get_scalar("nbands_sum"),
            "direct": self._get_data("direct"),
            "fundamental": self._get_data("fundamental"),
            "temperatures": self._get_data("temperatures"),
        }
        return _dict


class ElectronPhononBandgap(base.Refinery):
    @base.data_access
    def __str__(self):
        return "electron phonon bandgap"

    @base.data_access
    def to_dict(self, selection=None):
        return {
            "naccumulators": len(self._raw_data.valid_indices),
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
    def select(self, selection):
        """Return a list of ElectronPhononBandgapInstance objects matching the selection.

        Parameters
        ----------
        selection : dict
            Dictionary with keys as selection names (e.g., "nbands_sum", "selfen_approx", "selfen_delta")
            and values as the desired values for those properties.

        Returns
        -------
        list of ElectronPhononBandgapInstance
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
                selected_instances.append(ElectronPhononBandgapInstance(self, idx))
        return selected_instances

    @base.data_access
    def _get_data(self, name, index):
        return getattr(self._raw_data, name)[index][:]

    @base.data_access
    def _get_scalar(self, name, index):
        return getattr(self._raw_data, name)[index][()]

    @base.data_access
    def __getitem__(self, key):
        # TODO add logic to select instances
        return ElectronPhononBandgapInstance(self, key)

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
