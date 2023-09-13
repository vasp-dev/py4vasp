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
        set_appropriate_attrs(mlff_error_analysis)
        return mlff_error_analysis

    @classmethod
    def from_files(cls, dft_data, mlff_data):
        mlff_error_analysis = cls(_internal=True)
        calculations = Calculations.from_files(dft_data=dft_data, mlff_data=mlff_data)
        mlff_error_analysis._calculations = calculations
        set_appropriate_attrs(mlff_error_analysis)
        return mlff_error_analysis


def set_appropriate_attrs(cls):
    for datatype in ["dft", "mlff"]:
        set_energies(cls, tag="free energy    TOTEN", datatype=datatype)


def set_energies(cls, tag, datatype):
    all_energies = cls._calculations.energies.read()
    energy_data = all_energies[f"{datatype}_data"]
    energies = np.array([_energy_data[tag] for _energy_data in energy_data])
    setattr(cls, f"{datatype}_energies", energies)
