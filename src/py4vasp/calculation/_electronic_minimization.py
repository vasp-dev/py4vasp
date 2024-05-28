# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np

from py4vasp import exception, raw
from py4vasp._third_party import graph
from py4vasp.calculation import _base, _slice


class ElectronicMinimization(_slice.Mixin, _base.Refinery, graph.Mixin):
    """Access the convergence data for each electronic step.

    The OSZICAR file written out by VASP stores information related to convergence.
    Please check the vasp-wiki (https://www.vasp.at/wiki/index.php/OSZICAR) for more
    details about the exact outputs generated for each combination of INCAR tags."""

    def _more_than_one_ionic_step(self, data):
        return any(isinstance(_data, list) for _data in data) == True

    @_base.data_access
    def __str__(self):
        format_rep = "{0:g}\t{1:0.12E}\t{2:0.6E}\t{3:0.6E}\t{4:g}\t{5:0.3E}\t{6:0.3E}\n"
        label_rep = "{}\t\t{}\t\t{}\t\t{}\t\t{}\t{}\t\t{}\n"
        string = ""
        labels = [label.decode("utf-8") for label in getattr(self._raw_data, "label")]
        data = self.to_dict()
        electronic_iterations = data["N"]
        if not self._more_than_one_ionic_step(electronic_iterations):
            electronic_iterations = [electronic_iterations]
        ionic_steps = len(electronic_iterations)
        for ionic_step in range(ionic_steps):
            string += label_rep.format(*labels)
            electronic_steps = len(electronic_iterations[ionic_step])
            for electronic_step in range(electronic_steps):
                _data = []
                for label in self._raw_data.label:
                    _values_electronic = data[label.decode("utf-8")]
                    if not self._more_than_one_ionic_step(_values_electronic):
                        _values_electronic = [_values_electronic]
                    _value = _values_electronic[ionic_step][electronic_step]
                    _data.append(_value)
                _data = [float(_value) for _value in _data]
                string += format_rep.format(*_data)
        return string

    @_base.data_access
    def to_dict(self, selection=None):
        """Extract convergence data from the HDF5 file and make it available in a dict

        Parameters
        ----------
        selection: str
            Choose from either iteration_number, free_energy, free_energy_change,
            bandstructure_energy_change, number_hamiltonian_evaluations, norm_residual,
            difference_charge_density to get specific columns of the OSZICAR file. In
            case no selection is provided, supply all columns.

        Returns
        -------
        dict
            Contains a dict from the HDF5 related to OSZICAR convergence data
        """
        return_data = {}
        if selection is None:
            keys_to_include = self._from_bytes_to_utf(self._raw_data.label)
        else:
            labels_as_str = self._from_bytes_to_utf(self._raw_data.label)
            if selection not in labels_as_str:
                message = """\
Please choose a selection including at least one of the following keywords:
N, E, dE, deps, ncg, rms, rms(c)"""
                raise exception.RefinementError(message)
            keys_to_include = [selection]
        for key in keys_to_include:
            return_data[key] = self._read(key)
        return return_data

    def _from_bytes_to_utf(self, quantity: list):
        return [_quantity.decode("utf-8") for _quantity in quantity]

    @_base.data_access
    def _read(self, key):
        # data represents all of the electronic steps for all ionic steps
        data = getattr(self._raw_data, "convergence_data")
        iteration_number = data[:, 0]
        split_index = np.where(iteration_number == 1)[0]
        data = np.vsplit(data, split_index)[1:][self._steps]
        if isinstance(self._steps, slice):
            data = [raw.VaspData(_data) for _data in data]
        else:
            data = [raw.VaspData(data)]
        labels = [label.decode("utf-8") for label in self._raw_data.label]
        data_index = labels.index(key)
        return_data = [list(_data[:, data_index]) for _data in data]
        is_none = [_data.is_none() for _data in data]
        if len(return_data) == 1:
            return_data = return_data[0]
        return return_data if not np.all(is_none) else {}

    def to_graph(self, selection="E"):
        """Graph the change in parameter with iteration number.

        Parameters
        ----------
        selection: str
            Choose strings consistent with the OSZICAR format

        Returns
        -------
        Graph
            The Graph with the quantity plotted on y-axis and the iteration number of
            the x-axis.
        """
        data = self.to_dict()
        series = graph.Series(data["N"], data[selection], selection)
        ylabel = " ".join(select.capitalize() for select in selection.split("_"))
        return graph.Graph(
            series=[series],
            xlabel="Iteration number",
            ylabel=ylabel,
        )

    @_base.data_access
    def is_converged(self):
        is_elmin_converged = self._raw_data.is_elmin_converged[self._steps]
        converged = is_elmin_converged == 0
        if isinstance(converged, bool):
            converged = np.array([converged])
        return converged.flatten()
