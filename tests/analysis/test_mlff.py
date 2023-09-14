# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest

from py4vasp import Calculation, exception
from py4vasp._analysis.mlff import MLFFErrorAnalysis
from py4vasp.data import Energy, Force, Stress

from ..conftest import raw_data


class BaseCalculations:
    def read(self):
        return self._data


class Energies(BaseCalculations):
    def __init__(self, data):
        self._data = data


class Forces(BaseCalculations):
    def __init__(self, data):
        self._data = data


class Stresses(BaseCalculations):
    def __init__(self, data):
        self._data = data


class MockCalculations:
    @classmethod
    def set_attributes(
        cls,
        _energies: Dict[str, np.ndarray],
        _forces: Dict[str, np.ndarray],
        _stresses: Dict[str, np.ndarray],
    ):
        _cls = cls()
        setattr(_cls, "energies", Energies(data=_energies))
        setattr(_cls, "forces", Forces(data=_forces))
        setattr(_cls, "stresses", Stresses(data=_stresses))
        return _cls

    def number_of_calculations(self):
        return {"dft_data": 1, "mlff_data": 1}


@pytest.fixture
def mock_calculations(raw_data):
    data = defaultdict(lambda: defaultdict(list))
    for datatype in ["dft_data", "mlff_data"]:
        raw_energy = raw_data.energy("relax", randomize=True)
        energy = Energy.from_data(raw_energy)
        energy_data = energy.read()
        data["_energies"][datatype].append(energy_data)
        raw_force = raw_data.force("Sr2TiO4", randomize=True)
        force = Force.from_data(raw_force)
        force_data = force.read()
        data["_forces"][datatype].append(force_data)
        structure_data = force_data.pop("structure")
        force_data.update(structure_data)
        raw_stress = raw_data.stress("Sr2TiO4", randomize=True)
        stress = Stress.from_data(raw_stress)
        stress_data = stress.read()
        data["_stresses"][datatype].append(stress_data)
    data = {key: dict(value) for key, value in data.items()}
    _mock_calculations = MockCalculations.set_attributes(**data)
    return _mock_calculations


@pytest.fixture
def mock_calculations_incorrect(raw_data):
    data = defaultdict(lambda: defaultdict(list))
    for datatype in ["dft_data", "mlff_data"]:
        raw_energy = raw_data.energy("relax", randomize=True)
        energy = Energy.from_data(raw_energy)
        energy_data = energy.read()
        data["_energies"][datatype].append(energy_data)
        if datatype == "mlff_data":
            species = "Sr2TiO4"
        else:
            species = "Fe3O4"
        raw_force = raw_data.force(species, randomize=True)
        force = Force.from_data(raw_force)
        force_data = force.read()
        data["_forces"][datatype].append(force_data)
        structure_data = force_data.pop("structure")
        force_data.update(structure_data)
        raw_stress = raw_data.stress("Sr2TiO4", randomize=True)
        stress = Stress.from_data(raw_stress)
        stress_data = stress.read()
        data["_stresses"][datatype].append(stress_data)
    data = {key: dict(value) for key, value in data.items()}
    _mock_calculations = MockCalculations.set_attributes(**data)
    return _mock_calculations


@patch("py4vasp._data.base.Refinery.from_path", autospec=True)
@patch("py4vasp.raw.access", autospec=True)
def test_read_inputs_from_path(mock_access, mock_from_path):
    absolute_path_dft = Path(__file__) / "dft"
    absolute_path_mlff = Path(__file__) / "mlff"
    error_analysis = MLFFErrorAnalysis.from_paths(
        dft_data=absolute_path_dft, mlff_data=absolute_path_mlff
    )
    assert isinstance(error_analysis.mlff_energies, np.ndarray)
    assert isinstance(error_analysis.dft_energies, np.ndarray)
    assert isinstance(error_analysis.mlff_forces, np.ndarray)
    assert isinstance(error_analysis.dft_forces, np.ndarray)
    assert isinstance(error_analysis.mlff_lattice_vectors, np.ndarray)
    assert isinstance(error_analysis.dft_lattice_vectors, np.ndarray)
    assert isinstance(error_analysis.mlff_positions, np.ndarray)
    assert isinstance(error_analysis.dft_positions, np.ndarray)
    assert isinstance(error_analysis.mlff_nconfig, int)
    assert isinstance(error_analysis.dft_nconfig, int)
    assert isinstance(error_analysis.mlff_nions, np.ndarray)
    assert isinstance(error_analysis.dft_nions, np.ndarray)
    assert isinstance(error_analysis.mlff_stresses, np.ndarray)
    assert isinstance(error_analysis.dft_stresses, np.ndarray)


@patch("py4vasp._data.base.Refinery.from_path", autospec=True)
@patch("py4vasp.raw.access", autospec=True)
def test_read_inputs_from_files(mock_analysis, mock_from_path):
    absolute_files_dft = Path(__file__) / "dft*.h5"
    absolute_files_mlff = Path(__file__) / "mlff*.h5"
    error_analysis = MLFFErrorAnalysis.from_files(
        dft_data=absolute_files_dft, mlff_data=absolute_files_mlff
    )
    assert isinstance(error_analysis.mlff_energies, np.ndarray)
    assert isinstance(error_analysis.dft_energies, np.ndarray)
    assert isinstance(error_analysis.mlff_forces, np.ndarray)
    assert isinstance(error_analysis.dft_forces, np.ndarray)
    assert isinstance(error_analysis.mlff_lattice_vectors, np.ndarray)
    assert isinstance(error_analysis.dft_lattice_vectors, np.ndarray)
    assert isinstance(error_analysis.mlff_positions, np.ndarray)
    assert isinstance(error_analysis.dft_positions, np.ndarray)
    assert isinstance(error_analysis.mlff_nconfig, int)
    assert isinstance(error_analysis.dft_nconfig, int)
    assert isinstance(error_analysis.mlff_nions, np.ndarray)
    assert isinstance(error_analysis.dft_nions, np.ndarray)
    assert isinstance(error_analysis.mlff_stresses, np.ndarray)
    assert isinstance(error_analysis.dft_stresses, np.ndarray)


def test_read_from_data(mock_calculations):
    expected_energies = mock_calculations.energies.read()
    expected_forces = mock_calculations.forces.read()
    expected_stresses = mock_calculations.stresses.read()
    mlff_error_analysis = MLFFErrorAnalysis._from_data(mock_calculations)
    output_energies = mlff_error_analysis._calculations.energies.read()
    output_forces = mlff_error_analysis._calculations.forces.read()
    output_stresses = mlff_error_analysis._calculations.stresses.read()
    assert output_energies == expected_energies
    assert output_forces == expected_forces
    assert output_stresses == expected_stresses


def _iter_properties(tag, data):
    return np.array([_data[tag] for _data in data])


def test_attributes_from_data(mock_calculations):
    energies_dict = mock_calculations.energies.read()
    mlff_energies = _iter_properties("free energy    TOTEN", energies_dict["mlff_data"])
    dft_energies = _iter_properties("free energy    TOTEN", energies_dict["dft_data"])
    forces_dict = mock_calculations.forces.read()
    mlff_forces = _iter_properties("forces", forces_dict["mlff_data"])
    dft_forces = _iter_properties("forces", forces_dict["dft_data"])
    mlff_lattice_vectors = _iter_properties("lattice_vectors", forces_dict["mlff_data"])
    dft_lattice_vectors = _iter_properties("lattice_vectors", forces_dict["dft_data"])
    mlff_positions = _iter_properties("positions", forces_dict["mlff_data"])
    dft_positions = _iter_properties("positions", forces_dict["dft_data"])
    mlff_config, mlff_nions, _ = mlff_positions.shape
    dft_config, dft_nions, _ = dft_positions.shape
    stresses_dict = mock_calculations.stresses.read()
    mlff_stresses = _iter_properties("stress", stresses_dict["mlff_data"])
    dft_stresses = _iter_properties("stress", stresses_dict["dft_data"])
    mlff_error_analysis = MLFFErrorAnalysis._from_data(mock_calculations)
    assert np.array_equal(mlff_error_analysis.mlff_energies, mlff_energies)
    assert np.array_equal(mlff_error_analysis.dft_energies, dft_energies)
    assert np.array_equal(mlff_error_analysis.mlff_forces, mlff_forces)
    assert np.array_equal(mlff_error_analysis.dft_forces, dft_forces)
    assert np.array_equal(
        mlff_error_analysis.mlff_lattice_vectors, mlff_lattice_vectors
    )
    assert np.array_equal(mlff_error_analysis.dft_lattice_vectors, dft_lattice_vectors)
    assert np.array_equal(mlff_error_analysis.mlff_positions, mlff_positions)
    assert np.array_equal(mlff_error_analysis.dft_positions, dft_positions)
    assert mlff_error_analysis.mlff_nions == mlff_nions
    assert mlff_error_analysis.dft_nions == dft_nions
    assert mlff_error_analysis.mlff_nconfig == mlff_config
    assert mlff_error_analysis.dft_nconfig == dft_config
    assert np.array_equal(mlff_error_analysis.mlff_stresses, mlff_stresses)
    assert np.array_equal(mlff_error_analysis.dft_stresses, dft_stresses)


def test_validator(mock_calculations_incorrect):
    with pytest.raises(exception.IncorrectUsage):
        mlff_error_analysis = MLFFErrorAnalysis._from_data(mock_calculations_incorrect)


def test_energy_error_computation(mock_calculations):
    mlff_error_analysis = MLFFErrorAnalysis._from_data(mock_calculations)

    def _energy_error_per_atom(mlff_energy, dft_energy, natoms):
        return (dft_energy - mlff_energy) / natoms

    expected_energy_error = _energy_error_per_atom(
        mlff_energy=mlff_error_analysis.mlff_energies,
        dft_energy=mlff_error_analysis.dft_energies,
        natoms=mlff_error_analysis.mlff_nions,
    )
    output_energy_error = mlff_error_analysis.get_energy_error_per_atom()
    assert np.array_equal(expected_energy_error, output_energy_error)


def test_force_error_computation(mock_calculations):
    mlff_error_analysis = MLFFErrorAnalysis._from_data(mock_calculations)

    def rmse_error_force(mlff_force, dft_force, natoms):
        norm_error = np.linalg.norm(dft_force - mlff_force, axis=-1)
        rmse = np.sqrt(np.sum(norm_error**2, axis=-1) / (3 * natoms))
        return rmse

    expected_force_error = rmse_error_force(
        mlff_force=mlff_error_analysis.mlff_forces,
        dft_force=mlff_error_analysis.dft_forces,
        natoms=mlff_error_analysis.mlff_nions,
    )
    output_force_error = mlff_error_analysis.get_force_rmse()
    assert np.array_equal(expected_force_error, output_force_error)


def test_stress_error_computation(mock_calculations):
    mlff_error_analysis = MLFFErrorAnalysis._from_data(mock_calculations)

    def rmse_error_stress(mlff_stress, dft_stress, natoms):
        mlff_stress = np.triu(mlff_stress)
        dft_stress = np.triu(dft_stress)
        norm_error = np.linalg.norm(dft_stress - mlff_stress, axis=-1)
        rmse = np.sqrt(np.sum(norm_error**2, axis=-1) / (6 * natoms))
        return rmse

    expected_stress_error = rmse_error_stress(
        mlff_stress=mlff_error_analysis.mlff_stresses,
        dft_stress=mlff_error_analysis.dft_stresses,
        natoms=mlff_error_analysis.mlff_nions,
    )
    output_stress_error = mlff_error_analysis.get_stress_rmse()
    assert np.array_equal(expected_stress_error, output_stress_error)
