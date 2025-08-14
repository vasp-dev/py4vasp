# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp._calculation import base
from py4vasp._calculation.electron_phonon_accumulator import ElectronPhononAccumulator
from py4vasp._calculation.electron_phonon_chemical_potential import (
    ElectronPhononChemicalPotential,
)
from py4vasp._calculation.electron_phonon_instance import ElectronPhononInstance
from py4vasp._third_party import graph
from py4vasp._util import import_, index, select

pd = import_.optional("pandas")


class ElectronPhononTransportInstance(ElectronPhononInstance):
    """
    Represents a single instance of electron-phonon transport calculations.
    This class provides access to various transport properties computed from
    electron-phonon interactions, such as conductivity, mobility, Seebeck and Peltier
    coefficients, and thermal conductivity. It allows for data extraction, selection,
    and visualization of transport properties for a given calculation index.
    Parameters
    ----------
    parent : object
        The parent object containing the calculation data and methods for data retrieval.
    index : int
        The index identifying this particular transport calculation instance.
    """

    def __str__(self):
        """
        Returns a string representation of the transport instance, including chemical
        potential and scattering approximation information.
        """
        lines = []
        lines.append(f"Transport calculator N =  {self.index + 1}")
        lines.append("----------------------------------")
        # Information about the chemical potential
        mu_tag, mu_val = self.get_mu_tag()
        lines.append(f"{mu_tag}: {mu_val}")
        # Information about the scattering approximation
        scattering_approx = self._get_data("scattering_approximation")
        lines.append(f"scattering_approximation: {scattering_approx}")
        # Information about the broadening parameter
        delta = self._get_scalar("delta")
        lines.append(f"delta: {delta}")
        # Information about the number of bands summed over
        nbands_sumdelta = self._get_scalar("nbands_sum")
        lines.append(f"nbands_sum: {delta}")
        return "\n".join(lines) + "\n"

    def to_dict(self):
        """Returns a dictionary with selected transport properties for this instance."""
        names = [
            "temperatures",
            "transport_function",
            "electronic_conductivity",
            "mobility",
            "seebeck",
            "peltier",
            "electronic_thermal_conductivity",
        ]
        result = {name: self._get_data(name) for name in names}
        result["metadata"] = self._read_metadata()
        return result

    def selections(self):
        """Returns the available property names that can be selected for this instance."""
        return self.selections_units.keys()

    @property
    def selections_units(self):
        selections_units_dict = {
            "electronic_conductivity": "S/m",
            "mobility": "cm^2/(V.s)",
            "seebeck": "μV/K",
            "peltier": "μV",
            "electronic_thermal_conductivity": "W/(m.K)",
        }
        return selections_units_dict

    def _get_temperature_idx(self, temperature, tolerance=1e-8):

        def find_float_index(float_array, target_value, tolerance):
            close_indices = np.where(
                np.isclose(float_array, target_value, atol=tolerance)
            )[0]
            if close_indices.size > 0:
                return close_indices[0]
            else:
                raise ValueError(
                    f"No temperature close to {target_value} within a tolerance of {tolerance} was found."
                )

        return find_float_index(self._get_data("temperatures"), temperature, tolerance)

    def _get_ydata(self, selection):
        data_ = self._get_data(selection[0]).reshape([-1, 9])
        maps = {
            1: self._init_directions_dict(),
        }
        selector = index.Selector(maps, data_, reduction=np.average)
        return selector[selection[1:]]

    def _get_ydata_at_temperature(self, selection, temperature):
        itemp = self._get_temperature_idx(temperature)
        return self._get_ydata(selection)[itemp]

    def to_graph(self, selection):
        tree = select.Tree.from_selection(selection)
        series = []
        for selection in tree.selections():
            if selection[0] not in self.selections():
                raise ValueError(
                    f"Invalid selection {selection}. Must be one of {self.selections()}"
                )
            y = self._get_ydata(selection)
            x = self._get_data("temperatures")
            dir_str = "".join(selection[1:])
            series.append(graph.Series(x, y, label=f"{selection[0]} {dir_str}"))
        return graph.Graph(series, ylabel=selection[0], xlabel="Temperature (K)")

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
        return dict(zip(self.id_name, self.id_index - 1))


class ElectronPhononTransport(base.Refinery):
    """
    Provides access to electron-phonon transport data and selection utilities.
    This class serves as an interface to electron-phonon transport calculations,
    allowing users to query and select transport coefficients and related properties.
    It supports selection based on various criteria such as
    the number of bands, scattering approximation, broadening parameter (delta),
    and chemical potential.
    """

    def _accumulator(self):
        return ElectronPhononAccumulator(self, self._raw_data)

    @base.data_access
    def __str__(self):
        return str(self._accumulator())

    @base.data_access
    def to_dict(self):
        return self._accumulator().to_dict()

    @base.data_access
    def selections(self):
        """Return a dictionary describing what options are available
        to read the electron transport coefficients.
        This is done using the self-energy class."""
        base_selections = super().selections()
        return self._accumulator().selections(base_selections)

    @base.data_access
    def id_name(self):
        return self._raw_data.id_name[:]

    @base.data_access
    def id_size(self):
        return self._raw_data.id_size[:]

    @base.data_access
    def __getitem__(self, key):
        return ElectronPhononTransportInstance(self, key)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @base.data_access
    def _get_data(self, name, index):
        return self._accumulator().get_data(name, index)

    # @base.data_access
    # def _get_scalar(self, name, index):
    #     return getattr(self._raw_data, name)[index][()]

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
    def to_graph_carrier(self, selection, temperature):
        """
        Plot a particular transport coefficient as a function of the chemical potential tag
        for a particular temperature.
        """
        mu_tag, mu_val = self.chemical_potential_mu_tag()
        if selection == "":
            return None
        tree = select.Tree.from_selection(selection)
        series = []
        for selection in tree.selections():
            ydata = []
            for idx in range(len(self)):
                instance = self[idx]
                y = instance._get_ydata_at_temperature(selection, temperature)
                ydata.append(y)
            series.append(
                graph.Series(
                    mu_val, ydata, label=f"{selection[0]} {''.join(selection[1:])}"
                )
            )
        return graph.Graph(
            series,
            xlabel=mu_tag,
            ylabel=selection[0],
        )

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
                selected_instances.append(ElectronPhononTransportInstance(self, idx))
        return selected_instances
