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
        del data["metadata"]
        temperatures = data.pop("temperatures")
        fundamental_gap = data["fundamental"][:, np.newaxis]
        data["fundamental"] = fundamental_gap + data["fundamental_renorm"]
        data["direct"] = data["direct"][:, np.newaxis] + data["direct_renorm"]
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
            "metadata": {},
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
        # This class only make sense when the scattering approximation is SERTA
        return {
            **super().selections(),
            "scattering_approx": ("SERTA",),
            "carrier_per_cell": (),
            "carrier_den": (),
            "mu": (),
        }

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
        indices = self._select_indices(selection)
        return [ElectronPhononBandgapInstance(self, index) for index in indices]

    def _select_indices(self, selection):
        tree = select.Tree.from_selection(selection)
        return {
            index_
            for selection in tree.selections()
            for index_ in self._filter_indices(selection)
        }

    def _filter_indices(self, selection):

        # any of those
        # elph_scattering_approx -> scattering_approx
        # elph_nbands_sum        -> nbands_sum
        # elph_selfen_delta      -> selfen_delta

        # one of these
        # elph_selfen_carrier_den      -> selfen_carrier_den -> mu_tag
        # elph_selfen_carrier_per_cell -> selfen_carrier_per_cell -> mu_tag
        # elph_selfen_mu               -> selfen_mu -> mu_tag

        calc.electron_phonon.bandgap.select("scatter_approx=SERTA nbands_sum=300")

        # selection (Group(["nbands_sum", "12"], "="),) (Group(["selfen_approx", "SERTA"], "="),) # or
        # selection (Group(["nbands_sum", "12"], "="), Group(["selfen_approx", "SERTA"], "=")) # and
        remaining_indices = range(len(self))
        for group in selection:
            assert isinstance(group, select.Group)
            assert group.separator == "="
            assert len(group.group) == 2
            remaining_indices = self._filter_group(remaining_indices, group)
            remaining_indices = list(remaining_indices)
        yield from remaining_indices

    def _filter_group(self, remaining_indices, group):
        for index_ in remaining_indices:
            if self._match_key_value(index_, *group.group):
                yield index_

    def _match_key_value(self, index_, key, value):
        instance_value = self._get_data(key, index_)
        if value.isnumeric():
            return np.isclose(instance_value, float(value), atol=0)
        else:
            return instance_value == value

    @base.data_access
    def __getitem__(self, key):
        return ElectronPhononBandgapInstance(self, key)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @base.data_access
    def _get_data(self, name, index):
        if name == "carrier_den":
            _, mu_val = self.chemical_potential_mu_tag()
            return mu_val[index]
        dataset = getattr(self._raw_data, name)
        return np.array(dataset[index])[()]

    @base.data_access
    def __len__(self):
        return len(self._raw_data.valid_indices)
