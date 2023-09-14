# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import Calculations


class MLFFErrorAnalysis:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def _from_data(cls, _calculations):
        mlff_error_analysis = cls(_internal=True)
        mlff_error_analysis._calculations = _calculations
        set_appropriate_attrs(mlff_error_analysis)
        return mlff_error_analysis

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
        set_forces_related_attributes(cls, datatype=datatype)
        set_stresses(cls, datatype=datatype)


def set_energies(cls, tag, datatype):
    all_energies = cls._calculations.energies.read()
    energy_data = all_energies[f"{datatype}_data"]
    energies = np.array([_energy_data[tag] for _energy_data in energy_data])
    setattr(cls, f"{datatype}_energies", energies)


def set_forces_related_attributes(cls, datatype):
    all_force_data = cls._calculations.forces.read()
    force_data = all_force_data[f"{datatype}_data"]
    forces = np.array([_force_data["forces"] for _force_data in force_data])
    lattice_vectors = [_force_data["lattice_vectors"] for _force_data in force_data]
    lattice_vectors = np.array(lattice_vectors)
    positions = np.array([_force_data["positions"] for _force_data in force_data])
    nions = positions.shape[0]
    setattr(cls, f"{datatype}_forces", forces)
    setattr(cls, f"{datatype}_lattice_vectors", lattice_vectors)
    setattr(cls, f"{datatype}_positions", positions)
    setattr(cls, f"{datatype}_nions", nions)


def set_stresses(cls, datatype):
    all_stress_data = cls._calculations.stresses.read()
    stress_data = all_stress_data[f"{datatype}_data"]
    stresses = np.array([_stress_data["stress"] for _stress_data in stress_data])
    setattr(cls, f"{datatype}_stresses", stresses)
