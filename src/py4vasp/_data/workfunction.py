# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import data
from py4vasp._data import base
from py4vasp._third_party import graph


class Workfunction(base.Refinery, graph.Mixin):
    """The workfunction of a material describes the energy required to remove an electron
    to the vacuum.

    In VASP you can compute the workfunction by setting the IDIPOL flag in the INCAR file.
    This class provides then the functionality to analyze the resulting potential."""

    @base.data_access
    def to_dict(self):
        bandgap = data.Bandgap.from_data(self._raw_data.reference_potential)
        return {
            "direction": f"lattice vector {self._raw_data.idipol}",
            "distance": self._raw_data.distance[:],
            "average_potential": self._raw_data.average_potential[:],
            "vacuum_potential": self._raw_data.vacuum_potential[:],
            "valence_band_maximum": bandgap.valence_band_maximum(),
            "conduction_band_minimum": bandgap.conduction_band_minimum(),
            "fermi_energy": self._raw_data.fermi_energy,
        }

    @base.data_access
    def to_graph(self):
        data = self.to_dict()
        series = graph.Series(
            data["distance"], data["average_potential"], data["direction"]
        )
        return graph.Graph(
            series=series, xlabel="distance (Å)", ylabel="average potential (eV)"
        )
