# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation import base, slice_
from py4vasp._util import check, convert, import_


class ElectronPhononChemicalPotential(base.Refinery):
    "Placeholder for electron phonon chemical potential"

    @base.data_access
    def __str__(self):
        return "electron phonon chemical pontential"

    @base.data_access
    def mu_tag(self):
        """Get INCAR tag that was used to set the carrier density"""
        if not check.is_none(self._raw_data.carrier_den):
            return "selfen_carrier_den", self._raw_data.carrier_den[:]
        if not check.is_none(self._raw_data.mu):
            return "selfen_mu", self._raw_data.selfen_mu[:]
        if not check.is_none(self._raw_data.carrier_per_cell):
            return "selfen_carrier_per_cell", self._raw_data.carrier_per_cell[:]

    @base.data_access
    def to_dict(self):
        _dict = {
            "fermi_energy": self._raw_data.fermi_energy,
            "chemical_potential": self._raw_data.chemical_potential[:],
            "carrier_density": self._raw_data.carrier_density[:],
            "temperatures": self._raw_data.temperatures[:],
        }
        tag, value = self.mu_tag()
        _dict[tag] = value
        return _dict
