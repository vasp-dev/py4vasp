# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from collections import abc

import numpy as np

from py4vasp import raw
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    merge_strings,
    quantity,
)
from py4vasp._calculation.electron_phonon_accumulator import ElectronPhononAccumulator
from py4vasp._calculation.electron_phonon_instance import ElectronPhononInstance
from py4vasp._third_party import graph
from py4vasp._util import convert, index, select


class BandgapInstance(ElectronPhononInstance, graph.Mixin):
    """
    Represents an instance of electron-phonon band gap calculations.

    This class provides methods to access, and visualize the temperature-dependent
    direct and fundamental band gaps due to electron-phonon interactions.
    This is constructed with an index corresponding to the accumulator index
    as shown in the OUTCAR.

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
        quasi_particle_gaps = data[f"{label}_renorm"][spin]
        for temperature, quasi_particle_gap in zip(temperatures, quasi_particle_gaps):
            renormalization = quasi_particle_gap - kohn_sham_gap
            yield f"{temperature:18.6f} {kohn_sham_gap:19.6f} {quasi_particle_gap:19.6f} {1000 * renormalization:19.6f}"
        yield ""

    def to_graph(self, selection):
        """Generates a graph representing the temperature dependence of bandgap energies.

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
        data["fundamental"] = np.tile(data["fundamental"], (len(temperatures), 1)).T
        data["direct"] = np.tile(data["direct"], (len(temperatures), 1)).T
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

    def to_dict(self):
        """Convert the electron-phonon bandgap calculation results to a dictionary.

        Returns
        -------
        dict
            A dictionary containing:
            - "metadata": A dictionary with metadata about the calculation, including *nbands_sum* (the sum of the number of bands), *selfen_delta* (the self-energy delta value), and *<mu_tag>* (the chemical potential value for the current index).
            - "direct_renorm": The renormalized direct bandgap values.
            - "direct": The direct bandgap values.
            - "fundamental_renorm": The renormalized fundamental bandgap values.
            - "fundamental": The fundamental bandgap values.
            - "temperatures": The temperatures at which the calculations were performed.

        Notes
        -----
        The <mu_tag> key in the metadata will be dynamically set based on the chemical
        potential tag returned by `ChemicalPotential.mu_tag()`.
        """
        return {
            "metadata": self.read_metadata(),
            "direct_renorm": self._get_data("direct_renorm"),
            "direct": self._get_data("direct"),
            "fundamental_renorm": self._get_data("fundamental_renorm"),
            "fundamental": self._get_data("fundamental"),
            "temperatures": self._get_data("temperatures"),
        }


class ElectronPhononBandgapHandler(abc.Sequence):
    """Handler for electron-phonon bandgap data."""

    def __init__(self, raw_data: raw.ElectronPhononBandgap):
        self._raw_data = raw_data

    @classmethod
    def from_data(
        cls, raw_data: raw.ElectronPhononBandgap
    ) -> "ElectronPhononBandgapHandler":
        return cls(raw_data)

    def _accumulator(self):
        return ElectronPhononAccumulator(self, self._raw_data)

    def __str__(self):
        return str(self._accumulator())

    def to_dict(self):
        return self._accumulator().to_dict()

    def selections(self):
        base_selections = {
            convert.quantity_name(self.__class__.__name__.replace("Handler", "")): [
                "default"
            ],
        }
        result = self._accumulator().selections(base_selections)
        result.pop("scattering_approx", None)
        return result

    def chemical_potential_mu_tag(self):
        return self._accumulator().chemical_potential_mu_tag()

    def select(self, selection):
        indices = self._accumulator().select_indices(
            selection, scattering_approximation="SERTA"
        )
        return [BandgapInstance(self, index) for index in indices]

    def _get_data(self, name, index):
        return self._accumulator().get_data(name, index)

    def __getitem__(self, key):
        if 0 <= key < len(self):
            mask = np.equal(self._raw_data.scattering_approximation, "SERTA")
            index_ = np.arange(len(mask))[mask][key]
            return BandgapInstance(self, index_)
        raise IndexError("Index out of range for electron phonon bandgap instance.")

    def __len__(self):
        return sum(np.equal(self._raw_data.scattering_approximation, "SERTA"))


@quantity("bandgap", group="electron_phonon")
class ElectronPhononBandgap(abc.Sequence):
    """
    ElectronPhononBandgap provides access to the electron-phonon bandgap renormalization
    data and selection utilities.

    This class allows users to query and select specific instances of electron-phonon
    bandgap calculations, based on the INCAR settings that were used to generate them
    (e.g. nbands_sum, selfen_delta or <mu_tag>). It provides methods to convert the data
    to dictionary form, retrieve available selection options, and access individual
    bandgap instances.

    Notes
    -----
    The <mu_tag> key in the metadata will be dynamically set based on the chemical
    potential tag returned by `ChemicalPotential.mu_tag()`.
    """

    def __init__(self, source, quantity_name: str = "electron_phonon_bandgap"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_data: raw.ElectronPhononBandgap) -> "ElectronPhononBandgap":
        return cls(source=DataSource(raw_data))

    def _handler_factory(self, raw):
        return ElectronPhononBandgapHandler.from_data(raw)

    def __str__(self):
        return merge_strings(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            ElectronPhononBandgapHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def to_dict(self, selection=None):
        """
        Converts the bandgap data to a dictionary format.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            ElectronPhononBandgapHandler.to_dict,
        )

    def read(self, selection=None):
        return self.to_dict(selection=selection)

    def selections(self, selection=None):
        """Return a dictionary describing what options are available to read the
        electron transport coefficients.

        Returns
        -------
        dict
            A dictionary with keys as selection names and values as the corresponding
            values. The keys include:
            - "nbands_sum": The sum of the number of bands.
            - "selfen_delta": The self-energy delta value.
            - <mu_tag>: The chemical potential value for the current index.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            ElectronPhononBandgapHandler.selections,
        )

    def chemical_potential_mu_tag(self, selection=None):
        """
        Retrieves the INCAR tag that was used to set the chemical potential
        as well as its values.

        Returns
        -------
        tuple[str, numpy.ndarray]
            The INCAR tag name and its corresponding value as set in the calculation.
            Possible tags are 'selfen_carrier_den', 'selfen_mu', or 'selfen_carrier_per_cell'.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            ElectronPhononBandgapHandler.chemical_potential_mu_tag,
        )

    def select(self, selection):
        """Return a list of BandgapInstance objects matching the selection.

        Parameters
        ----------
        selection : str
            A string specifying which instances we would like to select. You specify a
            particular string like "nbands_sum=800" to select all instances that were
            run with that setup. If you provide multiple selections the results will be
            merged.

        Returns
        -------
        list[BandgapInstance]
            Instances that match the selection criteria.
        """
        with self._source.access(self._quantity_name) as raw:
            return self._handler_factory(raw).select(selection)

    def __getitem__(self, key):
        with self._source.access(self._quantity_name) as raw:
            return self._handler_factory(raw)[key]

    def __len__(self):
        with self._source.access(self._quantity_name) as raw:
            return len(self._handler_factory(raw))
