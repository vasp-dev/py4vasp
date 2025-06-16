# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from collections import abc

import numpy as np

from py4vasp import exception
from py4vasp._calculation import base
from py4vasp._calculation.electron_phonon_chemical_potential import (
    ElectronPhononChemicalPotential,
)
from py4vasp._third_party import graph
from py4vasp._util import index, select, suggest

ALIAS = {
    "selfen_delta": "delta",
    "scattering_approx": "scattering_approximation",
}


class ElectronPhononBandgapInstance(graph.Mixin):
    """
    Represents an instance of electron-phonon band gap calculations.

    This class provides methods to access, format, and visualize the temperature-dependent
    direct and fundamental band gaps, including their renormalizations due to electron-phonon
    interactions. It is typically constructed with a reference to a parent calculation object
    and an index identifying the specific dataset.

    Attributes:
        parent: Reference to the parent calculation object containing the data.
        index: Index specifying which dataset to access from the parent.
    """

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
        """
        Generates a graph representing the temperature dependence of bandgap energies.
        This method accesses the electron-phonon bandgap data, applies the specified selection,
        and returns a graph object with energy values (in eV) plotted against temperature (in K).
        The graph includes series for the fundamental and direct bandgaps, with and without
        electron-phonon renormalization, as determined by the selection.
        Parameters
        ----------
        selection : str or object
            A selection string or object specifying which bandgap data to include in the graph.
            The selection is parsed and used to extract the relevant data series.
        Returns
        -------
        graph.Graph
            A graph object containing the selected bandgap energy series as a function of temperature.
        """
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
        "Convenient wrapper around to_dict. Check that function for examples and optional arguments."
        return self.to_dict()

    def to_dict(self):
        """
        Convert the electron-phonon bandgap calculation results to a dictionary.
        Returns:
            dict: A dictionary containing:
                - "metadata": A dictionary with metadata about the calculation, including:
                    - "nbands_sum": The sum of the number of bands.
                    - "selfen_delta": The self-energy delta value.
                    - <mu_tag>: The chemical potential value for the current index.
                - "direct_renorm": The renormalized direct bandgap values.
                - "direct": The direct bandgap values.
                - "fundamental_renorm": The renormalized fundamental bandgap values.
                - "fundamental": The fundamental bandgap values.
                - "temperatures": The temperatures at which the calculations were performed.
        Notes:
            The <mu_tag> key in the metadata will be dynamically set based on the chemical potential tag
            returned by `ChemicalPotential.mu_tag()`.
        """

        mu_tag, mu_val = self.parent.chemical_potential_mu_tag()
        return {
            "metadata": {
                "nbands_sum": self._get_data("nbands_sum"),
                "selfen_delta": self._get_data("delta"),
                "scattering_approx": self._get_data("scattering_approximation"),
                mu_tag: mu_val[self.index],
            },
            "direct_renorm": self._get_data("direct_renorm"),
            "direct": self._get_data("direct"),
            "fundamental_renorm": self._get_data("fundamental_renorm"),
            "fundamental": self._get_data("fundamental"),
            "temperatures": self._get_data("temperatures"),
        }


class ElectronPhononBandgap(base.Refinery, abc.Sequence):
    """
    ElectronPhononBandgap provides access to the electron-phonon bandgap renormalization data
    and selection utilities.

    This class allows users to query and select specific instances of electron-phonon bandgap
    calculations, based on various selection criteria.
    It provides methods to convert the data to dictionary form, retrieve available selection
    options, and access individual bandgap instances.
    """

    @base.data_access
    def __str__(self):
        num_instances = len(self)
        selection_options = self.selections()
        options_str = "\n".join(
            f"  {key}: {value}" for key, value in selection_options.items()
        )
        return (
            f"ElectronPhononBandgap with {num_instances} instance(s).\n"
            f"Selection options:\n{options_str}"
        )

    @base.data_access
    def to_dict(self):
        """
        Converts the bandgap data to a dictionary format.
        """
        return {"naccumulators": len(self._raw_data.valid_indices)}

    @base.data_access
    def selections(self):
        """Return a dictionary describing what options are available
        to read the electron transport coefficients.
        This is done using the self-energy class."""
        # This class only make sense when the scattering approximation is SERTA
        mu_tag, mu_val = self.chemical_potential_mu_tag()
        return {
            **super().selections(),
            mu_tag: mu_val,
            "nbands_sum": self._raw_data.nbands_sum[:],
            "selfen_delta": self._raw_data.delta[:],
            "scattering_approx": self._raw_data.scattering_approximation[:],
        }

    @base.data_access
    def chemical_potential_mu_tag(self):
        """
        Retrieves the INCAR tag that was used to set the chemical potential
        as well as its values.

        Returns
        -------
        tuple of (str, numpy.ndarray)
            The INCAR tag name and its corresponding value as set in the calculation.
            Possible tags are 'selfen_carrier_den', 'selfen_mu', or 'selfen_carrier_per_cell'.
        """
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
        remaining_indices = range(len(self))
        for group in selection:
            self._raise_error_if_group_format_incorrect(group)
            assert len(group.group) == 2
            remaining_indices = self._filter_group(remaining_indices, *group.group)
            remaining_indices = list(remaining_indices)
        yield from remaining_indices

    def _raise_error_if_group_format_incorrect(self, group):
        if not isinstance(group, select.Group) or group.separator != "=":
            message = f'\
The selection {group} is not formatted correctly. It should be formatted like \
"key=value". Please check the "selections" method for available options.'
            raise exception.IncorrectUsage(message)

    def _filter_group(self, remaining_indices, key, value):
        for index_ in remaining_indices:
            if self._match_key_value(index_, key, str(value)):
                yield index_

    def _match_key_value(self, index_, key, value):
        instance_value = self._get_data(key, index_)
        try:
            value = float(value)
        except ValueError:
            return instance_value == value
        return np.isclose(instance_value, float(value), rtol=1e-8, atol=0)

    @base.data_access
    def __getitem__(self, key):
        if 0 <= key < len(self):
            return ElectronPhononBandgapInstance(self, key)
        raise IndexError("Index out of range for electron phonon bandgap instance.")

    @base.data_access
    def __len__(self):
        return len(self._raw_data.valid_indices)

    @base.data_access
    def _get_data(self, name, index):
        name = ALIAS.get(name, name)
        dataset = getattr(self._raw_data, name, None)
        if dataset is None:
            expected_name, dataset = self.chemical_potential_mu_tag()
            self._raise_error_if_not_present(name, expected_name)

        return np.array(dataset[index])[()]

    def _raise_error_if_not_present(self, name, expected_name):
        if name != expected_name:
            valid_names = set(self.selections().keys())
            valid_names.remove("electron_phonon_bandgap")
            did_you_mean = suggest.did_you_mean(name, valid_names)
            available_selections = '", "'.join(valid_names)
            message = f'\
The selection "{name}" is not a valid choice. {did_you_mean}Please check the \
available selections: "{available_selections}".'
            raise exception.IncorrectUsage(message)
