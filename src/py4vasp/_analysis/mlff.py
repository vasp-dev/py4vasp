# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from typing import Dict

import numpy as np

from py4vasp import Calculations, exception


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

    def get_energy_error_per_atom(self, normalize_by_configurations=False):
        error = (self.dft_energies - self.mlff_energies) / self.dft_nions
        if normalize_by_configurations:
            error = np.sum(np.abs(error), axis=-1) / self.dft_nconfig
        return error

    def _get_rmse(self, dft_quantity, mlff_quantity, degrees_of_freedom):
        norm_error = np.linalg.norm(dft_quantity - mlff_quantity, axis=-1)
        error = np.sqrt(np.sum(norm_error**2, axis=-1) / degrees_of_freedom)
        return error

    def get_force_rmse(self, normalize_by_configurations=False):
        deg_freedom = 3 * self.dft_nions
        error = self._get_rmse(self.dft_forces, self.mlff_forces, deg_freedom)
        if normalize_by_configurations:
            error = np.sum(error, axis=-1) / self.dft_nconfig
        return error

    def get_stress_rmse(self, normalize_by_configurations=False):
        deg_freedom = 6
        dft_stresses = np.triu(self.dft_stresses)
        mlff_stresses = np.triu(self.mlff_stresses)
        error = self._get_rmse(dft_stresses, mlff_stresses, deg_freedom)
        if normalize_by_configurations:
            error = np.sum(error, axis=-1) / self.dft_nconfig
        return error


def set_appropriate_attrs(cls):
    for datatype in ["dft", "mlff"]:
        set_accounting_attributes(cls, datatype=datatype)
        set_energies(cls, tag="free energy    TOTEN", datatype=datatype)
        set_forces_related_attributes(cls, datatype=datatype)
        set_stresses(cls, datatype=datatype)
    validate_data(cls)


def validate_data(cls):
    try:
        np.testing.assert_almost_equal(cls.dft_positions, cls.mlff_positions)
        np.testing.assert_almost_equal(
            cls.dft_lattice_vectors, cls.mlff_lattice_vectors
        )
        np.testing.assert_almost_equal(cls.dft_nions, cls.mlff_nions)
    except AssertionError:
        raise exception.IncorrectUsage(
            """\
Please pass a consistent set of data between DFT and MLFF calculations."""
        )


def set_accounting_attributes(cls, datatype):
    nconfig = cls._calculations.number_of_calculations()[f"{datatype}_data"]
    force_data = cls._calculations.forces.read()[f"{datatype}_data"]
    structures = _dict_to_list(force_data, "structure")
    elements = _dict_to_array(structures, "elements")
    nions = np.array([len(_elements) for _elements in elements])
    setattr(cls, f"{datatype}_nconfig", nconfig)
    setattr(cls, f"{datatype}_nions", nions)


def set_energies(cls, tag, datatype):
    all_energies = cls._calculations.energies.read()
    energy_data = all_energies[f"{datatype}_data"]
    energies = np.array([_energy_data[tag] for _energy_data in energy_data])
    setattr(cls, f"{datatype}_energies", energies)


def _dict_to_array(data: Dict, key: str) -> np.ndarray:
    return np.array([_data[key] for _data in data])


def _dict_to_list(data: Dict, key: str) -> list:
    return [_data[key] for _data in data]


def set_forces_related_attributes(cls, datatype):
    all_force_data = cls._calculations.forces.read()
    force_data = all_force_data[f"{datatype}_data"]
    forces = _dict_to_array(force_data, "forces")
    structures = _dict_to_list(force_data, "structure")
    lattice_vectors = _dict_to_array(structures, "lattice_vectors")
    lattice_vectors = np.array(lattice_vectors)
    positions = _dict_to_array(structures, "positions")
    setattr(cls, f"{datatype}_forces", forces)
    setattr(cls, f"{datatype}_lattice_vectors", lattice_vectors)
    setattr(cls, f"{datatype}_positions", positions)


def set_stresses(cls, datatype):
    all_stress_data = cls._calculations.stresses.read()
    stress_data = all_stress_data[f"{datatype}_data"]
    stresses = np.array([_stress_data["stress"] for _stress_data in stress_data])
    setattr(cls, f"{datatype}_stresses", stresses)
