# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from typing import Any, Dict, Tuple

from numpy.typing import NDArray

from py4vasp import exception, raw
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    merge_strings,
    quantity,
)
from py4vasp._util import check


class ElectronPhononChemicalPotentialHandler:
    """Handler for electron-phonon chemical potential data."""

    def __init__(self, raw_data: raw.ElectronPhononChemicalPotential):
        self._raw_data = raw_data

    @classmethod
    def from_data(
        cls, raw_data: raw.ElectronPhononChemicalPotential
    ) -> "ElectronPhononChemicalPotentialHandler":
        return cls(raw_data)

    def __str__(self) -> str:
        return "\n".join(self._generate_lines())

    def _generate_lines(self):
        temperatures = self._raw_data.temperatures[:]
        carrier_densities = self._raw_data.carrier_density[:].T
        chemical_potentials = self._raw_data.chemical_potential[:].T
        underline = "-" * 28
        format_row = lambda values: "".join(f"{x:14.8f}" for x in values)

        yield " " * 19 + "Number of electrons per cell"
        yield " " * 19 + underline
        for T, chemical_potential in zip(temperatures, chemical_potentials):
            yield f"T={T:16.8f}" + format_row(chemical_potential)
        yield " " * 19 + underline
        yield " " * 23 + "Chemical potential"
        yield " " * 19 + underline
        for T, carrier_density in zip(temperatures, carrier_densities):
            yield f"T={T:16.8f}" + format_row(carrier_density)
        yield " " * 19 + underline

    def mu_tag(self) -> Tuple[str, NDArray]:
        """
        Get the INCAR tag and value used to set the carrier density or chemical potential.

        Returns
        -------
        -
            The INCAR tag name and its corresponding value as set in the calculation.
            Possible tags are 'selfen_carrier_den', 'selfen_mu', or 'selfen_carrier_per_cell'.
        """
        if not check.is_none(self._raw_data.carrier_den):
            return "selfen_carrier_den", self._raw_data.carrier_den[:]
        if not check.is_none(self._raw_data.mu):
            return "selfen_mu", self._raw_data.mu[:]
        if not check.is_none(self._raw_data.carrier_per_cell):
            return "selfen_carrier_per_cell", self._raw_data.carrier_per_cell[:]
        raise exception.NoData(
            "None of the carrier density, chemical potential, or carrier per cell data is available in the raw data."
        )

    def label(self) -> str:
        """
        Get a descriptive label for the electron-phonon chemical potential data.

        Returns
        -------
        -
            A label indicating the type of data contained in this object and its units.
        """
        if not check.is_none(self._raw_data.carrier_den):
            return "Carrier density (cm^-3)"
        if not check.is_none(self._raw_data.mu):
            return "Chemical potential (eV)"
        if not check.is_none(self._raw_data.carrier_per_cell):
            return "Carrier per cell"
        raise exception.NoData(
            "None of the carrier density, chemical potential, or carrier per cell data is available in the raw data."
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the electron-phonon chemical potential data to a dictionary.

        Returns
        -------
        -
            A dictionary containing the Fermi energy, chemical potential, carrier density,
            temperatures, and the INCAR tag/value used to set the carrier density.
        """
        tag, value = self.mu_tag()
        return {
            "fermi_energy": self._raw_data.fermi_energy,
            "chemical_potential": self._raw_data.chemical_potential[:],
            "carrier_density": self._raw_data.carrier_density[:],
            "temperatures": self._raw_data.temperatures[:],
            tag: value,
        }


@quantity("chemical_potential", group="electron_phonon")
class ElectronPhononChemicalPotential:
    """
    Provides access to the electron-phonon chemical potential data calculated
    during an electron-phonon calculation.

    This class allows users to retrieve information about the chemical potential,
    carrier density, Fermi energy, and related quantities as computed in electron-phonon
    calculations. It also provides access to the INCAR tag used to set the carrier density.
    """

    def __init__(
        self, source, quantity_name: str = "electron_phonon_chemical_potential"
    ):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(
        cls, raw_data: raw.ElectronPhononChemicalPotential
    ) -> "ElectronPhononChemicalPotential":
        return cls(source=DataSource(raw_data))

    def _handler_factory(self, raw):
        return ElectronPhononChemicalPotentialHandler.from_data(raw)

    def __str__(self, selection=None) -> str:
        """
        Return a formatted string representation of the electron-phonon chemical potential object.
        """
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            ElectronPhononChemicalPotentialHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self) -> Dict[str, Any]:
        """
        Convert the electron-phonon chemical potential data to a dictionary.

        Returns
        -------
        -
            A dictionary containing the Fermi energy, chemical potential, carrier density,
            temperatures, and the INCAR tag/value used to set the carrier density.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            ElectronPhononChemicalPotentialHandler.to_dict,
        )

    def to_dict(self, selection=None) -> Dict[str, Any]:
        """Convenient alias for :py:meth:`read`. Please read the documentation there."""
        return self.read()

    def mu_tag(self) -> Tuple[str, NDArray]:
        """
        Get the INCAR tag and value used to set the carrier density or chemical potential.

        Returns
        -------
        -
            The INCAR tag name and its corresponding value as set in the calculation.
            Possible tags are 'selfen_carrier_den', 'selfen_mu', or 'selfen_carrier_per_cell'.

        Notes
        -----
        The method checks for the presence of carrier density, chemical potential,
        or carrier per cell in the raw data and returns the first one found.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            ElectronPhononChemicalPotentialHandler.mu_tag,
        )

    def label(self) -> str:
        """
        Get a descriptive label for the electron-phonon chemical potential data.

        This can be useful for plotting or identifying the type of data.

        Returns
        -------
        -
            A label indicating the type of data contained in this object and its units.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            ElectronPhononChemicalPotentialHandler.label,
        )
