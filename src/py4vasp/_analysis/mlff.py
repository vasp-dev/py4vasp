# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import Calculations


class MLFFErrorAnalysis:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_paths(cls, dft_data, mlff_data):
        mlff_error_analysis = cls(_internal=True)
        calculations = Calculations.from_paths(dft_data=dft_data, mlff_data=mlff_data)
        mlff_error_analysis._calculations = calculations
        return mlff_error_analysis

    @classmethod
    def from_files(cls, dft_data, mlff_data):
        mlff_error_analysis = cls(_internal=True)
        calculations = Calculations.from_files(dft_data=dft_data, mlff_data=mlff_data)
        mlff_error_analysis._calculations = calculations
        return mlff_error_analysis

    def mlff_energies(self):
        energy_data = self._calculations.energies.read()
        mlff_energy_data = energy_data["mlff_data"]
        tag = "free energy    TOTEN"
        energies = [_mlff_data[tag] for _mlff_data in mlff_energy_data]
        return np.array(energies)

    def dft_energies(self):
        energy_data = self._calculations.energies.read()
        dft_energy_data = energy_data["dft_data"]
        tag = "free energy    TOTEN"
        energies = [_dft_data[tag] for _dft_data in dft_energy_data]
        return np.array(energies)
