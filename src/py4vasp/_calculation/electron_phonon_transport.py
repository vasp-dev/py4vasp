# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from collections import abc
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import ArrayLike

from py4vasp import exception
from py4vasp._calculation import base
from py4vasp._calculation.electron_phonon_accumulator import ElectronPhononAccumulator
from py4vasp._calculation.electron_phonon_chemical_potential import (
    ElectronPhononChemicalPotential,
)
from py4vasp._calculation.electron_phonon_instance import ElectronPhononInstance
from py4vasp._third_party import graph
from py4vasp._util import import_, index, select

pd = import_.optional("pandas")

DIRECTIONS = {
    None: [0, 4, 8],
    "isotropic": [0, 4, 8],
    "xx": 0,
    "yy": 4,
    "zz": 8,
    "xy": [1, 3],
    "xz": [2, 6],
    "yz": [5, 7],
}


UNITS = {
    "electronic_conductivity": "S/m",
    "mobility": "cm^2/(V.s)",
    "seebeck": "μV/K",
    "peltier": "μV",
    "electronic_thermal_conductivity": "W/(m.K)",
}


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
        return f"Electron-phonon transport instance {self.index + 1}:\n{self._metadata_string()}"

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

    def temperatures(self):
        return self._get_data("temperatures")

    def electronic_conductivity(self, selection=""):
        return self._select_data("electronic_conductivity", selection)

    def mobility(self, selection=""):
        return self._select_data("mobility", selection)

    def seebeck(self, selection=""):
        return self._select_data("seebeck", selection)

    def peltier(self, selection=""):
        return self._select_data("peltier", selection)

    def electronic_thermal_conductivity(self, selection=""):
        return self._select_data("electronic_thermal_conductivity", selection)

    def _select_data(self, quantity, selection):
        tree = select.Tree.from_selection(selection)
        selections = list(tree.selections())
        maps = {1: DIRECTIONS}
        data = self._get_data(quantity).reshape(-1, 9)
        selector = index.Selector(maps, data, reduction=np.average)
        return {
            selector.label(selection): selector[selection] for selection in selections
        }

    def selections(self):
        """Returns the available property names that can be selected for this instance."""
        return UNITS.keys()

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


class ElectronPhononTransport(base.Refinery, abc.Sequence, graph.Mixin):
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
        """Return a dictionary that lists how many accumulators are available

        Returns
        -------
        dict
            Dictionary containing information about the available accumulators.
        """
        return self._accumulator().to_dict()

    @base.data_access
    def selections(self):
        """Return a dictionary describing what options are available to read the transport.

        Returns
        -------
        dict
            Dictionary containing available selection options with their possible values.
            Keys include selection criteria like "nbands_sum", "selfen_approx", "selfen_delta".
        """
        base_selections = {
            **super().selections(),
            "transport": list(UNITS.keys()),
        }
        return self._accumulator().selections(base_selections)

    @property
    def units(self):
        return UNITS

    @base.data_access
    def chemical_potential_mu_tag(self):
        chemical_potential = ElectronPhononChemicalPotential.from_data(
            self._raw_data.chemical_potential
        )
        return chemical_potential.mu_tag()

    @base.data_access
    def _get_data(self, name, index):
        return self._accumulator().get_data(name, index)

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
        return self._select_instances(selection)

    def _select_instances(self, selection, filter_keys=()):
        indices = self._accumulator().select_indices(selection, *filter_keys)
        return [ElectronPhononTransportInstance(self, index) for index in indices]

    @base.data_access
    def to_graph(self, selection):
        """
        Plot a particular transport coefficient as a function of the chemical potential tag
        for a particular temperature.
        """
        builder = SeriesBuilder()
        series_list = [
            series
            for selection in select.Tree.from_selection(selection).selections()
            for series in builder.build(selection, self._get_instances(selection))
        ]
        return graph.Graph(series_list)

    def _get_instances(self, selection):
        selection_string = select.selections_to_string((selection,))
        filter_keys = UNITS.keys() | DIRECTIONS.keys() | {"T", "temperature"}
        return self._select_instances(selection_string, filter_keys)

    @base.data_access
    def __getitem__(self, key):
        if 0 <= key < len(self._raw_data.valid_indices):
            return ElectronPhononTransportInstance(self, key)
        raise IndexError("Index out of range for electron-phonon transport instance.")

    @base.data_access
    def __len__(self):
        return len(self._raw_data.valid_indices)


class SeriesBuilder:
    def __init__(self):
        self.quantity = None

    def build(self, selection, instances):
        self.quantity = self._get_and_check_quantity(selection)
        x, annotations = self._get_metadata(instances)
        temperatures, mask = self._get_temperature(selection, instances)
        common_label, data = self._get_transport_data(selection, instances, mask)
        marker = self._use_marker_if_metadata_is_different(annotations)
        assert len(temperatures) == len(data)
        for T, y in zip(temperatures, data):
            label = f"{common_label}(T={T}K)"
            yield graph.Series(x, y, label, annotations=annotations, marker=marker)

    def _get_and_check_quantity(self, selection):
        selected_quantities = self._get_quantities_from_selection(selection)
        self._raise_error_if_not_exactly_one_quantity(selection, selected_quantities)
        self._raise_error_if_quantity_inconsistent(selected_quantities)
        return selected_quantities[0]

    def _get_quantities_from_selection(self, selection):
        return [
            quantity
            for quantity in UNITS.keys()
            if select.contains(selection, quantity)
        ]

    def _raise_error_if_quantity_inconsistent(self, selected_quantities):
        if self.quantity and self.quantity != selected_quantities[0]:
            raise exception.IncorrectUsage(
                f"Selections must contain exactly one transport quantity, but got {self.quantity} and {selected_quantities[0]}"
            )

    def _raise_error_if_not_exactly_one_quantity(self, selection, selected_quantities):
        if len(selected_quantities) != 1:
            raise exception.IncorrectUsage(
                f"Selection must contain exactly one transport quantity, but '{select.selections_to_string((selection,))}' contains {selected_quantities}."
            )

    def _get_metadata(self, instances):
        chemical_potential = np.empty(len(instances))
        nbands_sum = np.empty(len(instances), dtype=int)
        selfen_delta = np.empty(len(instances))
        scattering_approx = np.empty(len(instances), dtype="<U20")
        for ii, instance in enumerate(instances):
            metadata = instance._read_metadata()
            nbands_sum[ii] = metadata.pop("nbands_sum")
            selfen_delta[ii] = metadata.pop("selfen_delta")
            scattering_approx[ii] = metadata.pop("scattering_approx")
            _, chemical_potential[ii] = metadata.popitem()
        return chemical_potential, {
            "nbands_sum": nbands_sum,
            "selfen_delta": selfen_delta,
            "scattering_approx": scattering_approx,
        }

    def _get_temperature(self, selection, instances):
        selected_temperatures = self._get_temperature_from_selection(selection)
        all_temperatures = self._get_temperature_from_instances(instances)
        mask = self._find_selected_temperature(selected_temperatures, all_temperatures)
        return all_temperatures[mask], mask

    def _get_temperature_from_selection(self, selection):
        return [
            float(item.right_operand[0])
            for item in selection
            if isinstance(item, select.Assignment)
            and item.left_operand in {"T", "temperature"}
        ]

    def _get_temperature_from_instances(self, instances):
        all_temperatures = instances[0].temperatures()
        for instance in instances:
            assert np.allclose(all_temperatures, instance.temperatures())
        return all_temperatures

    def _find_selected_temperature(self, selected_temperatures, all_temperatures):
        if selected_temperatures:
            return np.isclose(all_temperatures, selected_temperatures[0])
        else:
            return np.full_like(all_temperatures, True, dtype=bool)

    def _get_transport_data(self, selection, instances, mask):
        selected_directions = self._get_direction_from_selection(selection)
        self._raise_error_if_more_than_one_direction(selection, selected_directions)
        direction = selected_directions[0] if selected_directions else None
        common_label = self._assign_common_label(direction)
        data = self._get_data_from_instances(instances, mask, direction)
        return common_label, data

    def _get_direction_from_selection(self, selection):
        return [
            direction
            for direction in DIRECTIONS.keys()
            if select.contains(selection, direction)
        ]

    def _raise_error_if_more_than_one_direction(self, selection, selected_directions):
        if len(selected_directions) > 1:
            raise exception.IncorrectUsage(
                f"Selection must contain exactly one transport direction, but '{select.selections_to_string((selection,))}' contains {selected_directions}."
            )

    def _assign_common_label(self, direction):
        if direction in ("isotropic", None):
            return self.quantity
        else:
            return f"{self.quantity}_{direction}"

    def _get_data_from_instances(self, instances, mask, direction):
        joint_data = []
        for instance in instances:
            get_quantity = getattr(instance, self.quantity)
            transport_data = get_quantity(selection=direction)
            assert len(transport_data) == 1
            _, value = transport_data.popitem()
            joint_data.append(value[mask])
        return np.array(joint_data).T

    def _use_marker_if_metadata_is_different(self, annotations):
        for value in annotations.values():
            if len(np.unique(value)) > 1:
                return "*"
        return None
