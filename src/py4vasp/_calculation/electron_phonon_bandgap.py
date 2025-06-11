# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp._calculation import base
from py4vasp._calculation.electron_phonon_chemical_potential import (
    ElectronPhononChemicalPotential,
)

# from py4vasp._calculation.electron_phonon_self_energy import ElectronPhononSelfEnergy
from py4vasp._third_party import graph
from py4vasp._util import index, select


class ElectronPhononBandgapInstance(graph.Mixin):
    "Placeholder for electron phonon band gap"

    def __init__(self, parent, index):
        self.parent = parent
        self.index = index

    def __str__(self):
        """
        Returns a formatted string representation of the band gap instance,
        including direct and fundamental gaps as a function of temperature.
        """
        return "\n".join(self._generate_lines())

    def _generate_lines(self):
        data = self.to_dict()
        num_component = len(data["fundamental"])
        for component in range(num_component):
            yield from self._format_spin_component(component, num_component)
            yield from self._format_gap_section("direct", component, data)
            yield from self._format_gap_section("fundamental", component, data)

    def _format_spin_component(self, component, num_component):
        if component == 0 and num_component == 3:
            yield "spin independent"
        elif num_component == 3:
            yield f"spin component {component}"
        yield ""

    def _format_gap_section(self, label, spin, data):
        yield f"{label.capitalize()} gap:"
        yield "   Temperature (K)         KS gap (eV)         QP gap (eV)     KS-QP gap (meV)"
        temperatures = data["temperatures"]
        kohn_sham_gap = data[label][spin]
        renormalizations = data[f"{label}_renorm"][spin]
        for temperature, renormalization in zip(temperatures, renormalizations):
            quasi_particle_gap = kohn_sham_gap + renormalization
            yield f"{temperature:18.6f} {kohn_sham_gap:19.6f} {quasi_particle_gap:19.6f} {1000 * renormalization:19.6f}"
        yield ""

    def print(self):
        "Print a string representation of this instance."
        print(str(self))

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def _get_data(self, name):
        return self.parent._get_data(name, self.index)

    def to_graph(self, selection):
        data = self.to_dict()
        temperatures = data.pop("temperatures")
        del data["nbands_sum"]
        data["fundamental"] = data["fundamental"] + data["fundamental_renorm"]
        data["direct"] = data["direct"] + data["direct_renorm"]
        maps = {0: {key: index_ for index_, key in enumerate(data.keys())}}
        selector = index.Selector(maps, np.array(list(data.values())))
        tree = select.Tree.from_selection(selection)
        series = [
            graph.Series(
                temperatures, selector[selection], label=selector.label(selection)
            )
            for selection in tree.selections()
        ]
        return graph.Graph(series, ylabel="Energy (eV)", xlabel="Temperature (K)")

    def read(self):
        return self.to_dict()

    def to_dict(self):
        return {
            "nbands_sum": self._get_data("nbands_sum"),
            "direct_renorm": self._get_data("direct_renorm"),
            "direct": self._get_data("direct"),
            "fundamental_renorm": self._get_data("fundamental_renorm"),
            "fundamental": self._get_data("fundamental"),
            "temperatures": self._get_data("temperatures"),
        }

    @property
    def id_index(self):
        return self._get_data("id_index")

    @property
    def id_name(self):
        return self.parent.id_name()


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
        # TODO: fix the use of self_energy
        return super().selections()
        self_energy = ElectronPhononSelfEnergy.from_data(self._raw_data.self_energy)
        selections = self_energy.selections()
        # This class only make sense when the scattering approximation is SERTA
        selections["selfen_approx"] = ["SERTA"]
        return selections

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
                        instance_value = self._get_data("nbands_sum", idx)
                        match_this = instance_value == value
                    elif key == "selfen_approx":
                        instance_value = self._get_data("scattering_approximation", idx)
                        match_this = instance_value == value
                    elif key == "selfen_delta":
                        instance_value = self._get_data("delta", idx)
                        match_this = abs(instance_value - value) < 1e-8
                    elif key == mu_tag:
                        mu_idx = self[idx].id_index[2] - 1
                        instance_value = mu_val[mu_idx]
                        match_this = abs(instance_value - float(value)) < 1e-8
                    else:
                        possible_values = self.selections()
                        raise ValueError(
                            f"Invalid selection {key}. Possible values are {possible_values.keys()}"
                        )
                    match = match and match_this
                match_all = match_all or match
            if match_all:
                selected_instances.append(ElectronPhononBandgapInstance(self, idx))
        return selected_instances

    @base.data_access
    def __getitem__(self, key):
        # TODO add logic to select instances
        return ElectronPhononBandgapInstance(self, key)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @base.data_access
    def _get_data(self, name, index):
        dataset = getattr(self._raw_data, name)
        return np.array(dataset[index])[()]

    @base.data_access
    def __len__(self):
        return len(self._raw_data.valid_indices)
