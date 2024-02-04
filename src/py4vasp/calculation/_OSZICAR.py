# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np

from py4vasp import exception, raw
from py4vasp._third_party import graph
from py4vasp._util import convert
from py4vasp.calculation import _base, _slice, _structure

INDEXING_OSZICAR = {
    "iteration_number": 0,
    "free_energy": 1,
    "free_energy_change": 2,
    "bandstructure_energy_change": 3,
    "number_hamiltonian_evaluations": 4,
    "norm_residual": 5,
    "difference_charge_density": 6,
}


class OSZICAR(_slice.Mixin, _base.Refinery, graph.Mixin):
    """Access the convergence data for each electronic step."""

    @_base.data_access
    def to_dict(self, selection=None):
        return_data = {}
        if selection is None:
            keys_to_include = INDEXING_OSZICAR
        else:
            if keys_to_include not in INDEXING_OSZICAR:
                message = """\
Please choose a selection including at least one of the following keywords:
iteration_number, free_energy, free_energy_change, bandstructure_energy_change,
number_hamiltonian_evaluations, norm_residual, difference_charge_density. Else do not
select anything and all OSZICAR outputs will be provided."""
                raise exception.RefinementError(message)
            keys_to_include = selection
        for key in INDEXING_OSZICAR:
            return_data[key] = self._read(key)
        return return_data

    def _read(self, key):
        # data represents all of the electronic steps for all ionic steps
        data = getattr(self._raw_data, "convergence_data")
        iteration_number = data[:, 0]
        split_index = np.where(iteration_number == 1)[0]
        data = np.vsplit(data, split_index)[1:][self._steps]
        data = raw.VaspData(data)
        data_index = INDEXING_OSZICAR[key]
        return data[:, data_index] if not data.is_none() else {}

    def to_graph(self, selection="free_energy"):
        data = self.to_dict()
        series = graph.Series(data["iteration_number"], data[selection], selection)
        return graph.Graph(
            series=[series], xlabel="Iteration number", ylabel="Free energy [eV]"
        )
