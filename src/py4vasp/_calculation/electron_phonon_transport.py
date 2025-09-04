# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from collections import abc
from typing import Any, Dict, Generator, List, Tuple

import numpy as np

from py4vasp import exception
from py4vasp._calculation import base
from py4vasp._calculation.electron_phonon_accumulator import ElectronPhononAccumulator
from py4vasp._calculation.electron_phonon_instance import ElectronPhononInstance
from py4vasp._third_party import graph
from py4vasp._util import index, select

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


class ElectronPhononTransportInstance(ElectronPhononInstance, graph.Mixin):
    """
    Represents a single instance of electron-phonon transport calculations.

    This class provides access to various transport properties computed from
    electron-phonon interactions, such as conductivity, mobility, Seebeck and Peltier
    coefficients, and thermal conductivity. It allows for data extraction, selection,
    and visualization of transport properties for a given calculation index.
    """

    def __str__(self):
        """
        Returns a string representation of the transport instance, including chemical
        potential and scattering approximation information.
        """
        return f"Electron-phonon transport instance {self.index + 1}:\n{self._metadata_string()}"

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary with selected transport properties for this instance.

        Returns
        -------
        A dictionary containing:
        - "metadata": Metadata about the instance, including chemical potential,
            number of bands summed, delta, and scattering approximation.
        - "temperatures": Array of temperatures at which transport properties are computed.
        - "transport_function": The transport function data.
        - "electronic_conductivity": Electronic conductivity values.
        - "mobility": Mobility values.
        - "seebeck": Seebeck coefficient values.
        - "peltier": Peltier coefficient values.
        - "electronic_thermal_conductivity": Electronic thermal conductivity values.
        """
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

    def temperatures(self) -> np.ndarray:
        """Returns the temperatures at which transport properties are computed.

        Returns
        -------
        A numpy array of temperatures in Kelvin.
        """
        return self._get_data("temperatures")

    def electronic_conductivity(
        self, selection=""
    ) -> np.ndarray | Dict[str, np.ndarray]:
        """Returns the electronic conductivity for the selected direction.

        Parameters
        ----------
        selection
            A string specifying the direction for which to retrieve the electronic
            conductivity. Options include "xx", "yy", "zz", "xy", "xz", "yz", or
            "isotropic". If no direction is specified, the isotropic average is returned.

        Returns
        -------
        A numpy array of electronic conductivity values in S/m for the specified
        direction, or a dictionary of arrays if multiple directions are selected.

        Examples
        --------
        To get the isotropic average of the electronic conductivity of the first instance:

        >>> calculation.electron_phonon.transport[0].electronic_conductivity()

        To get the electronic conductivity in the xx direction of all instances

        >>> [
        ...     instance.electronic_conductivity("xx")
        ...     for instance in calculation.electron_phonon.transport
        ... ]
        """
        return self._select_data("electronic_conductivity", selection)

    def mobility(self, selection="") -> np.ndarray | Dict[str, np.ndarray]:
        """Returns the mobility for the selected direction.

        Parameters
        ----------
        selection
            A string specifying the direction for which to retrieve the mobility. Options
            include "xx", "yy", "zz", "xy", "xz", "yz", or "isotropic". If no direction
            is specified, the isotropic average is returned.

        Returns
        -------
        A numpy array of mobility values in cm^2/(V.s) for the specified direction,
        or a dictionary of arrays if multiple directions are selected.

        Examples
        --------
        To get the isotropic average of the mobility of the first instance:

        >>> calculation.electron_phonon.transport[0].mobility()

        To get the mobility in the xx direction of all instances

        >>> [
        ...     instance.mobility("xx")
        ...     for instance in calculation.electron_phonon.transport
        ... ]
        """
        return self._select_data("mobility", selection)

    def seebeck(self, selection="") -> np.ndarray | Dict[str, np.ndarray]:
        """Returns the Seebeck coefficient for the selected direction.

        Parameters
        ----------
        selection
            A string specifying the direction for which to retrieve the Seebeck
            coefficient. Options include "xx", "yy", "zz", "xy", "xz", "yz", or
            "isotropic". If no direction is specified, the isotropic average is returned.

        Returns
        -------
        A numpy array of Seebeck coefficient values in μV/K for the specified direction,
        or a dictionary of arrays if multiple directions are selected.

        Examples
        --------
        To get the isotropic average of the Seebeck coefficient of the first instance:

        >>> calculation.electron_phonon.transport[0].seebeck()

        To get the Seebeck coefficient in the xx direction of all instances

        >>> [
        ...     instance.seebeck("xx")
        ...     for instance in calculation.electron_phonon.transport
        ... ]
        """
        return self._select_data("seebeck", selection)

    def peltier(self, selection="") -> np.ndarray | Dict[str, np.ndarray]:
        """Returns the Peltier coefficient for the selected direction.

        Parameters
        ----------
        selection
            A string specifying the direction for which to retrieve the Peltier
            coefficient. Options include "xx", "yy", "zz", "xy", "xz", "yz", or
            "isotropic". If no direction is specified, the isotropic average is returned.

        Returns
        -------
        A numpy array of Peltier coefficient values in μV for the specified direction,
        or a dictionary of arrays if multiple directions are selected.

        Examples
        --------
        To get the isotropic average of the Peltier coefficient of the first instance:

        >>> calculation.electron_phonon.transport[0].peltier()

        To get the Peltier coefficient in the xx direction of all instances

        >>> [
        ...     instance.peltier("xx")
        ...     for instance in calculation.electron_phonon.transport
        ... ]
        """
        return self._select_data("peltier", selection)

    def electronic_thermal_conductivity(
        self, selection=""
    ) -> np.ndarray | Dict[str, np.ndarray]:
        """Returns the electronic thermal conductivity for the selected direction.

        Parameters
        ----------
        selection
            A string specifying the direction for which to retrieve the electronic
            thermal conductivity. Options include "xx", "yy", "zz", "xy", "xz", "yz",
            or "isotropic". If no direction is specified, the isotropic average is returned.

        Returns
        -------
        A numpy array of electronic thermal conductivity values in W/(m.K) for the
        specified direction, or a dictionary of arrays if multiple directions are selected.

        Examples
        --------
        To get the isotropic average of the electronic thermal conductivity of the
        first instance:

        >>> calculation.electron_phonon.transport[0].electronic_thermal_conductivity()

        To get the electronic thermal conductivity in the xx direction of all instances

        >>> [
        ...     instance.electronic_thermal_conductivity("xx")
        ...     for instance in calculation.electron_phonon.transport
        ]
        """
        return self._select_data("electronic_thermal_conductivity", selection)

    def _select_data(self, quantity, selection):
        tree = select.Tree.from_selection(selection)
        selections = list(tree.selections())
        maps = {1: DIRECTIONS}
        data = self._get_data(quantity).reshape(-1, 9)
        selector = index.Selector(maps, data, reduction=np.average)
        result = {
            selector.label(selection): selector[selection] for selection in selections
        }
        return result if len(result) > 1 else next(iter(result.values()))

    def selections(self):
        """Returns the available property names that can be selected for this instance."""
        return UNITS.keys()

    def to_graph(self, selection):
        builder = _SeriesBuilderInstance()
        tree = select.Tree.from_selection(selection)
        series = [builder.build(selection, self) for selection in tree.selections()]
        return graph.Graph(series, xlabel="Temperature (K)", ylabel=builder.ylabel)


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
    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary that lists how many accumulators are available

        Returns
        -------
        Dictionary containing information about the available accumulators.
        """
        return self._accumulator().to_dict()

    @base.data_access
    def selections(self) -> Dict[str, Any]:
        """Return a dictionary describing what options are available to read the transport.

        Returns
        -------
        Dictionary containing available selection options with their possible values.
        Keys include selection criteria like "nbands_sum", "selfen_approx", "selfen_delta".
        """
        base_selections = {
            **super().selections(),
            "transport": list(UNITS.keys()),
        }
        return self._accumulator().selections(base_selections)

    @property
    def units(self) -> Dict[str, str]:
        """Return a dictionary with the physical units for each transport quantity.

        Returns
        -------
        Dictionary containing transport quantities as keys and their corresponding
        physical units as values.
        """
        return UNITS

    @base.data_access
    def chemical_potential_mu_tag(self) -> tuple[str, np.ndarray]:
        """Retrieves the INCAR tag that was used to set the chemical potential as well
        as its values.

        Returns
        -------
        The INCAR tag name and its corresponding value as set in the calculation.
        Possible tags are 'selfen_carrier_den', 'selfen_mu', or 'selfen_carrier_per_cell'.
        """
        return self._accumulator().chemical_potential_mu_tag()

    @base.data_access
    def _get_data(self, name, index):
        return self._accumulator().get_data(name, index)

    @base.data_access
    def select(self, selection: str) -> List[ElectronPhononTransportInstance]:
        """Return a list of ElectronPhononSelfEnergyInstance objects matching the selection.

        Parameters
        ----------
        selection
            A string specifying which instances we would like to select. You specify a
            particular string like "nbands_sum=800" to select all instances that were
            run with that setup. If you provide multiple selections the results will be
            merged.

        Returns
        -------
        Instances that match the selection criteria.

        Examples
        --------
        To select all instances with a sum of 800 bands, you can use:

        >>> calculation.electron_phonon.transport.select("nbands_sum=800")

        To select instances with a specific scattering approximation, such as SERTA:

        >>> calculation.electron_phonon.transport.select("selfen_approx=SERTA")

        You can also combine multiple selection criteria. For example, to select instances
        with a sum of 800 bands and a delta value of 0.1:

        >>> calculation.electron_phonon.transport.select("nbands_sum=800(selfen_delta=0.1)")
        """
        return self._select_instances(selection)

    def _select_instances(self, selection, filter_keys=()):
        indices = self._accumulator().select_indices(selection, *filter_keys)
        return [ElectronPhononTransportInstance(self, index) for index in indices]

    @base.data_access
    def to_graph(self, selection: str) -> graph.Graph:
        """
        Plot a particular transport coefficient as a function of the chemical potential tag.

        Parameters
        ----------
        selection
            Use this string to specify what you want to plot. You must always specify
            a transport quantity like "mobility" or "seebeck". You can optionally also
            specify a direction like "xx" or "isotropic". If you do not specify a
            direction, the isotropic average will be used. You can also specify a
            particular temperature by adding "T=300" to the selection. If you do not
            specify a temperature, results for all temperatures will be plotted.
            Finally, you can also filter the instances that are used for plotting by
            adding criteria like "nbands_sum=800" or "selfen_delta=0.1".

        Returns
        -------
        A graph object containing the requested data. Each series corresponds to one
        temperature and plots the requested transport quantity as a function of the
        chemical potential tag. If the instances share the same metadata, the series
        is connected with a line.

        Examples
        --------
        To plot the mobility as a function of the chemical potential tag for all
        available temperatures, you can use:

        >>> calculation.electron_phonon.transport.to_graph("mobility")

        To plot the Seebeck coefficient in the xx direction at a specific temperature
        of 300K, you can use:

        >>> calculation.electron_phonon.transport.to_graph("seebeck(xx(T=300))")

        You can also filter the instances used for plotting. For example, to plot the
        electronic conductivity for instances with a sum of 800 bands and a delta
        value of 0.1, you can use:

        >>> calculation.electron_phonon.transport.to_graph("electronic_conductivity(nbands_sum=800(selfen_delta=0.1))")
        """
        builder = _SeriesBuilderMapping()
        series_list = [
            series
            for selection in select.Tree.from_selection(selection).selections()
            for series in builder.build(selection, self._get_instances(selection))
        ]
        xlabel = self._accumulator().chemical_potential_label()
        return graph.Graph(series_list, xlabel=xlabel, ylabel=builder.ylabel)

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


class _SeriesBuilderBase:
    def __init__(self):
        self.quantity = None

    @property
    def ylabel(self) -> str:
        """Return a label for the y-axis based on the selected quantity."""
        quantity = self.quantity.replace("_", " ").capitalize()
        return f"{quantity} ({UNITS[self.quantity]})"

    def _get_and_check_quantity(self, selection):
        selected_quantities = self._get_quantities_from_selection(selection)
        self._raise_error_if_not_exactly_one_quantity(selection, selected_quantities)
        self._raise_error_if_quantity_inconsistent(selected_quantities)
        return selected_quantities[0]

    def _raise_error_if_quantity_inconsistent(self, selected_quantities):
        if self.quantity and self.quantity != selected_quantities[0]:
            raise exception.IncorrectUsage(
                f"Selections must contain exactly one transport quantity, but got {self.quantity} and {selected_quantities[0]}"
            )

    def _get_quantities_from_selection(self, selection):
        return [
            quantity
            for quantity in UNITS.keys()
            if select.contains(selection, quantity)
        ]

    def _raise_error_if_not_exactly_one_quantity(self, selection, selected_quantities):
        if len(selected_quantities) != 1:
            raise exception.IncorrectUsage(
                f"Selection must contain exactly one transport quantity, but '{select.selections_to_string((selection,))}' contains {selected_quantities}."
            )

    def _get_and_check_direction(self, selection):
        selected_directions = self._get_direction_from_selection(selection)
        self._raise_error_if_more_than_one_direction(selection, selected_directions)
        return selected_directions[0] if selected_directions else None

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

    def _get_data_from_instance(self, direction, instance):
        get_quantity = getattr(instance, self.quantity)
        return get_quantity(selection=direction)


class _SeriesBuilderInstance(_SeriesBuilderBase):
    def build(
        self, selection: Tuple, instance: ElectronPhononTransportInstance
    ) -> graph.Series:
        """Build a graph series for a single instance based on the selection.

        This will plot selected quantity over temperature."""
        self.quantity = self._get_and_check_quantity(selection)
        direction = self._get_and_check_direction(selection)
        x = instance.temperatures()
        y = self._get_data_from_instance(direction, instance)
        label = direction or "isotropic"
        return graph.Series(x, y, label=label)


class _SeriesBuilderMapping(_SeriesBuilderBase):
    def build(
        self, selection: Tuple, instances: List[ElectronPhononTransportInstance]
    ) -> Generator[graph.Series, None, None]:
        """Build graph series for multiple instances based on the selection.
        This will plot selected quantity over chemical potential tag for each temperature.
        """
        self.quantity = self._get_and_check_quantity(selection)
        direction = self._get_and_check_direction(selection)
        x, annotations = self._get_metadata(instances)
        temperatures, mask = self._get_temperature(selection, instances)
        data = self._get_transport_data(direction, instances, mask)
        common_label = self._assign_common_label(direction)
        marker = self._use_marker_if_metadata_is_different(annotations)
        assert len(temperatures) == len(data)
        for T, y in zip(temperatures, data):
            label = f"{common_label}T={T}K"
            yield graph.Series(x, y, label, annotations=annotations, marker=marker)

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
            float(item.right_operand)
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

    def _get_transport_data(self, direction, instances, mask):
        joint_data = []
        for instance in instances:
            transport_data = self._get_data_from_instance(direction, instance)
            joint_data.append(transport_data[mask])
        return np.array(joint_data).T

    def _assign_common_label(self, direction):
        if direction in ("isotropic", None):
            return ""
        else:
            return f"{direction}, "

    def _use_marker_if_metadata_is_different(self, annotations):
        for value in annotations.values():
            if len(np.unique(value)) > 1:
                return "*"
        return None
