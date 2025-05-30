# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation import base, slice_
from py4vasp._util import check, convert, import_


class ElectronPhononChemicalPotential(base.Refinery):
    """
    Provides access to the electron-phonon chemical potential data calculatedduring an
    electron-phonon calculation.

    This class allows users to retrieve information about the chemical potential,
    carrier density, Fermi energy, and related quantities as computed in electron-phonon
    calculations. It also provides access to the INCAR tag used to set the carrier density.
    """

    @base.data_access
    def __str__(self):
        """
        Return a formatted string representation of the electron-phonon chemical potential object.
        """
        # Extract data
        temps = self._raw_data.temperatures[:].T
        carrier_density = self._raw_data.carrier_density[:].T
        chemical_potential = self._raw_data.chemical_potential[:].T

        # Helper for formatting rows
        def format_row(T, values):
            return f"T={T:16.8f}" + "".join(f"{v:14.8f}" for v in values)

        # Build carrier density section
        lines = []
        lines.append("                   Number of electrons per cell")
        lines.append("                   ----------------------------")
        for i, T in enumerate(temps):
            lines.append(format_row(T, carrier_density[i]))
        lines.append("                   ----------------------------")
        lines.append("                       Chemical potential")
        lines.append("                   ----------------------------")
        for i, T in enumerate(temps):
            lines.append(format_row(T, chemical_potential[i]))
        lines.append("                   ----------------------------")
        return "\n".join(lines)

    @base.data_access
    def mu_tag(self):
        """
        Get the INCAR tag and value used to set the carrier density or chemical potential.

        Returns
        -------
        tuple of (str, numpy.ndarray)
            The INCAR tag name and its corresponding value as set in the calculation.
            Possible tags are 'selfen_carrier_den', 'selfen_mu', or 'selfen_carrier_per_cell'.

        Notes
        -----
        The method checks for the presence of carrier density, chemical potential,
        or carrier per cell in the raw data and returns the first one found.
        """
        if not check.is_none(self._raw_data.carrier_den):
            return "selfen_carrier_den", self._raw_data.carrier_den[:]
        if not check.is_none(self._raw_data.mu):
            return "selfen_mu", self._raw_data.selfen_mu[:]
        if not check.is_none(self._raw_data.carrier_per_cell):
            return "selfen_carrier_per_cell", self._raw_data.carrier_per_cell[:]

    @base.data_access
    def to_dict(self):
        """
        Convert the electron-phonon chemical potential data to a dictionary.

        Returns
        -------
        dict
            A dictionary containing the Fermi energy, chemical potential, carrier density,
            temperatures, and the INCAR tag/value used to set the carrier density.
        """
        _dict = {
            "fermi_energy": self._raw_data.fermi_energy,
            "chemical_potential": self._raw_data.chemical_potential[:],
            "carrier_density": self._raw_data.carrier_density[:],
            "temperatures": self._raw_data.temperatures[:],
        }
        tag, value = self.mu_tag()
        _dict[tag] = value
        return _dict
