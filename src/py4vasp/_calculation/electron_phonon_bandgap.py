# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation import base
from py4vasp._calculation.electron_phonon_chemical_potential import (
    ElectronPhononChemicalPotential,
)

# from py4vasp._calculation.electron_phonon_self_energy import ElectronPhononSelfEnergy
from py4vasp._third_party import graph
from py4vasp._util import select


class ElectronPhononBandgapInstance:
    "Placeholder for electron phonon band gap"

    def __init__(self, parent, index):
        self.parent = parent
        self.index = index

    def __str__(self):
        """
        Returns a formatted string representation of the band gap instance,
        including direct and fundamental gaps as a function of temperature.
        """
        lines = []
        lines.append(f"Electron self-energy accumulator N=  {self.index + 1}")

        def format_gap_section(title, ks_gap, qp_gap, temperatures):
            section = []
            section.append(f"{title}")
            section.append(
                "   Temperature (K)         KS gap (eV)         QP gap (eV)     KS-QP gap (meV)"
            )
            for t, qp in zip(temperatures, qp_gap):
                diff_meV = (ks_gap - qp) * 1000
                section.append(f"{t:18.6f}{ks_gap:20.6f}{qp:20.6f}{diff_meV:20.6f}")
            return "\n".join(section)

        # Get data
        temperatures = self._get_data("temperatures")
        ks_gap_direct = self._get_data("direct")
        qp_gap_direct = self._get_data("direct_renorm")
        ks_gap_fundamental = self._get_data("fundamental")
        qp_gap_fundamental = self._get_data("fundamental_renorm")

        nspin, ntemps = qp_gap_direct.shape
        for ispin in range(nspin):
            if nspin == 2:
                lines.append("spin independent")
            # Direct gap section
            lines.append("")
            lines.append(
                format_gap_section(
                    "Direct gap",
                    ks_gap_direct[ispin],
                    qp_gap_direct[ispin],
                    temperatures,
                )
            )
            lines.append("")
            # Fundamental gap section
            lines.append(
                format_gap_section(
                    "Fundamental gap",
                    ks_gap_fundamental[ispin],
                    qp_gap_fundamental[ispin],
                    temperatures,
                )
            )
            lines.append("")

        if nspin == 2:
            for ispin in range(nspin):
                if nspin == 2:
                    lines.append("spin component ", ispin + 1)
                # Direct gap section
                lines.append("")
                lines.append(
                    format_gap_section(
                        "Direct gap",
                        ks_gap_direct[ispin],
                        qp_gap_direct[ispin],
                        temperatures,
                    )
                )
                lines.append("")
                # Fundamental gap section
                lines.append(
                    format_gap_section(
                        "Fundamental gap",
                        ks_gap_fundamental[ispin],
                        qp_gap_fundamental[ispin],
                        temperatures,
                    )
                )
                lines.append("")
        return "\n".join(lines)

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

    def read(self):
        return self.to_dict()

    def to_dict(self):
        _dict = {
            "nbands_sum": self._get_scalar("nbands_sum"),
            "direct_renorm": self._get_data("direct_renorm"),
            "direct": self._get_scalar("direct"),
            "fundamental_renorm": self._get_data("fundamental_renorm"),
            "fundamental": self._get_scalar("fundamental"),
            "temperatures": self._get_data("temperatures"),
        }
        return _dict

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
                        instance_value = self._get_scalar("nbands_sum", idx)
                        match_this = instance_value == value
                    elif key == "selfen_approx":
                        instance_value = self._get_data("scattering_approximation", idx)
                        match_this = instance_value == value
                    elif key == "selfen_delta":
                        instance_value = self._get_scalar("delta", idx)
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
