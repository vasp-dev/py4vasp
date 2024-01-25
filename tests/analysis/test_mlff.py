# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from collections import defaultdict
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import numpy as np
import pytest

from py4vasp import exception, calculation
from py4vasp._analysis.mlff import MLFFErrorAnalysis


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
        _paths: Dict[str, Path],
    ):
        _cls = cls()
        setattr(_cls, "energies", Energies(data=_energies))
        setattr(_cls, "forces", Forces(data=_forces))
        setattr(_cls, "stresses", Stresses(data=_stresses))
        setattr(_cls, "_paths", _paths)
        return _cls

    def number_of_calculations(self):
        num_ions = len(self.forces.read()["mlff_data"])
        return {"dft_data": num_ions, "mlff_data": num_ions}

    def paths(self):
        return self._paths


@pytest.fixture
def mock_calculations(raw_data):
    data = defaultdict(lambda: defaultdict(list))
    for datatype in ["dft_data", "mlff_data"]:
        raw_energy = raw_data.energy("relax", randomize=True)
        energy = calculation.energy.from_data(raw_energy)
        energy_data = energy.read()
        data["_energies"][datatype].append(energy_data)
        raw_force = raw_data.force("Sr2TiO4", randomize=True)
        force = calculation.force.from_data(raw_force)
        force_data = force.read()
        data["_forces"][datatype].append(force_data)
        raw_stress = raw_data.stress("Sr2TiO4", randomize=True)
        stress = calculation.stress.from_data(raw_stress)
        stress_data = stress.read()
        data["_stresses"][datatype].append(stress_data)
        data["_paths"][datatype].append(Path(__file__) / "calc")
    data = {key: dict(value) for key, value in data.items()}
    _mock_calculations = MockCalculations.set_attributes(**data)
    return _mock_calculations


@pytest.fixture
def mock_multiple_calculations(raw_data):
    data = defaultdict(lambda: defaultdict(list))
    for datatype in ["dft_data", "mlff_data"]:
        for i in range(4):
            raw_energy = raw_data.energy("relax", randomize=True)
            energy = calculation.energy.from_data(raw_energy)
            energy_data = energy.read()
            data["_energies"][datatype].append(energy_data)
            raw_force = raw_data.force("Sr2TiO4", randomize=True)
            force = calculation.force.from_data(raw_force)
            force_data = force.read()
            data["_forces"][datatype].append(force_data)
            raw_stress = raw_data.stress("Sr2TiO4", randomize=True)
            stress = calculation.stress.from_data(raw_stress)
            stress_data = stress.read()
            data["_stresses"][datatype].append(stress_data)
            data["_paths"][datatype].append(Path(__file__) / "calc")
    data = {key: dict(value) for key, value in data.items()}
    _mock_calculations = MockCalculations.set_attributes(**data)
    return _mock_calculations


@pytest.fixture
def mock_calculations_incorrect(raw_data):
    data = defaultdict(lambda: defaultdict(list))
    for datatype in ["dft_data", "mlff_data"]:
        raw_energy = raw_data.energy("relax", randomize=True)
        energy = calculation.energy.from_data(raw_energy)
        energy_data = energy.read()
        data["_energies"][datatype].append(energy_data)
        if datatype == "mlff_data":
            species = "Sr2TiO4"
        else:
            species = "Fe3O4"
        raw_force = raw_data.force(species, randomize=True)
        force = calculation.force.from_data(raw_force)
        force_data = force.read()
        data["_forces"][datatype].append(force_data)
        raw_stress = raw_data.stress("Sr2TiO4", randomize=True)
        stress = calculation.stress.from_data(raw_stress)
        stress_data = stress.read()
        data["_stresses"][datatype].append(stress_data)
        data["_paths"][datatype].append(Path(__file__) / "calc")
    data = {key: dict(value) for key, value in data.items()}
    _mock_calculations = MockCalculations.set_attributes(**data)
    return _mock_calculations


@patch("py4vasp.calculation._base.Refinery.from_path", autospec=True)
@patch("py4vasp.raw.access", autospec=True)
def test_read_inputs_from_path(mock_access, mock_from_path):
    absolute_path_dft = Path(__file__) / "dft"
    absolute_path_mlff = Path(__file__) / "mlff"
    error_analysis = MLFFErrorAnalysis.from_paths(
        dft_data=absolute_path_dft, mlff_data=absolute_path_mlff
    )
    assert isinstance(error_analysis.mlff.energies, np.ndarray)
    assert isinstance(error_analysis.dft.energies, np.ndarray)
    assert isinstance(error_analysis.mlff.forces, np.ndarray)
    assert isinstance(error_analysis.dft.forces, np.ndarray)
    assert isinstance(error_analysis.mlff.lattice_vectors, np.ndarray)
    assert isinstance(error_analysis.dft.lattice_vectors, np.ndarray)
    assert isinstance(error_analysis.mlff.positions, np.ndarray)
    assert isinstance(error_analysis.dft.positions, np.ndarray)
    assert isinstance(error_analysis.mlff.nconfig, int)
    assert isinstance(error_analysis.dft.nconfig, int)
    assert isinstance(error_analysis.mlff.nions, np.ndarray)
    assert isinstance(error_analysis.dft.nions, np.ndarray)
    assert isinstance(error_analysis.mlff.stresses, np.ndarray)
    assert isinstance(error_analysis.dft.stresses, np.ndarray)


@patch("py4vasp.calculation._base.Refinery.from_path", autospec=True)
@patch("py4vasp.raw.access", autospec=True)
def test_read_inputs_from_files(mock_analysis, mock_from_path):
    absolute_files_dft = Path(__file__) / "dft*.h5"
    absolute_files_mlff = Path(__file__) / "mlff*.h5"
    error_analysis = MLFFErrorAnalysis.from_files(
        dft_data=absolute_files_dft, mlff_data=absolute_files_mlff
    )
    assert isinstance(error_analysis.mlff.energies, np.ndarray)
    assert isinstance(error_analysis.dft.energies, np.ndarray)
    assert isinstance(error_analysis.mlff.forces, np.ndarray)
    assert isinstance(error_analysis.dft.forces, np.ndarray)
    assert isinstance(error_analysis.mlff.lattice_vectors, np.ndarray)
    assert isinstance(error_analysis.dft.lattice_vectors, np.ndarray)
    assert isinstance(error_analysis.mlff.positions, np.ndarray)
    assert isinstance(error_analysis.dft.positions, np.ndarray)
    assert isinstance(error_analysis.mlff.nconfig, int)
    assert isinstance(error_analysis.dft.nconfig, int)
    assert isinstance(error_analysis.mlff.nions, np.ndarray)
    assert isinstance(error_analysis.dft.nions, np.ndarray)
    assert isinstance(error_analysis.mlff.stresses, np.ndarray)
    assert isinstance(error_analysis.dft.stresses, np.ndarray)


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


def _iter_properties(tag, data, return_array=True):
    if return_array:
        return np.array([_data[tag] for _data in data])
    else:
        return [_data[tag] for _data in data]


@pytest.mark.parametrize("mocker", ["mock_calculations", "mock_multiple_calculations"])
def test_attributes_from_data(mocker, request):
    mock_calculations = request.getfixturevalue(mocker)
    energies_dict = mock_calculations.energies.read()
    mlff_energies = _iter_properties("free energy    TOTEN", energies_dict["mlff_data"])
    dft_energies = _iter_properties("free energy    TOTEN", energies_dict["dft_data"])
    forces_dict = mock_calculations.forces.read()
    mlff_forces = _iter_properties("forces", forces_dict["mlff_data"])
    dft_forces = _iter_properties("forces", forces_dict["dft_data"])
    mlff_structures = _iter_properties(
        "structure", forces_dict["mlff_data"], return_array=False
    )
    dft_structures = _iter_properties(
        "structure", forces_dict["dft_data"], return_array=False
    )
    mlff_lattice_vectors = _iter_properties("lattice_vectors", mlff_structures)
    dft_lattice_vectors = _iter_properties("lattice_vectors", dft_structures)
    mlff_positions = _iter_properties("positions", mlff_structures)
    dft_positions = _iter_properties("positions", dft_structures)
    mlff_config, mlff_nions, _ = mlff_positions.shape
    dft_config, dft_nions, _ = dft_positions.shape
    stresses_dict = mock_calculations.stresses.read()
    mlff_stresses = _iter_properties("stress", stresses_dict["mlff_data"])
    dft_stresses = _iter_properties("stress", stresses_dict["dft_data"])
    mlff_error_analysis = MLFFErrorAnalysis._from_data(mock_calculations)
    assert np.array_equal(mlff_error_analysis.mlff.energies, mlff_energies)
    assert np.array_equal(mlff_error_analysis.dft.energies, dft_energies)
    assert np.array_equal(mlff_error_analysis.mlff.forces, mlff_forces)
    assert np.array_equal(mlff_error_analysis.dft.forces, dft_forces)
    assert np.array_equal(
        mlff_error_analysis.mlff.lattice_vectors, mlff_lattice_vectors
    )
    assert np.array_equal(mlff_error_analysis.dft.lattice_vectors, dft_lattice_vectors)
    assert np.array_equal(mlff_error_analysis.mlff.positions, mlff_positions)
    assert np.array_equal(mlff_error_analysis.dft.positions, dft_positions)
    assert mlff_error_analysis.mlff.nconfig == mlff_config
    assert mlff_error_analysis.dft.nconfig == dft_config
    assert np.array_equal(mlff_error_analysis.mlff.stresses, mlff_stresses)
    assert np.array_equal(mlff_error_analysis.dft.stresses, dft_stresses)


def test_validator(mock_calculations_incorrect):
    with pytest.raises(exception.IncorrectUsage):
        mlff_error_analysis = MLFFErrorAnalysis._from_data(mock_calculations_incorrect)


def test_energy_per_atom_computation(mock_calculations):
    mlff_error_analysis = MLFFErrorAnalysis._from_data(mock_calculations)

    def rmse_error_energy(mlff_energy, dft_energy, natoms):
        error = (mlff_energy - dft_energy) / natoms
        return error

    expected_energy_error = rmse_error_energy(
        mlff_energy=mlff_error_analysis.mlff.energies,
        dft_energy=mlff_error_analysis.dft.energies,
        natoms=mlff_error_analysis.mlff.nions,
    )
    output_energy_error = mlff_error_analysis.get_energy_error_per_atom()
    assert np.array_equal(expected_energy_error, output_energy_error)


def test_multiple_energy_per_atom_computation(mock_multiple_calculations):
    mlff_error_analysis = MLFFErrorAnalysis._from_data(mock_multiple_calculations)

    def rmse_error_energy(mlff_energy, dft_energy, natoms, nconfig):
        error = (mlff_energy - dft_energy) / natoms
        return np.sum(np.abs(error), axis=-1) / nconfig

    expected_energy_error = rmse_error_energy(
        mlff_energy=mlff_error_analysis.mlff.energies,
        dft_energy=mlff_error_analysis.dft.energies,
        natoms=mlff_error_analysis.mlff.nions,
        nconfig=mlff_error_analysis.mlff.nconfig,
    )
    output_energy_error = mlff_error_analysis.get_energy_error_per_atom(
        normalize_by_configurations=True
    )
    assert np.array_equal(expected_energy_error, output_energy_error)


def test_force_error_computation(mock_calculations):
    mlff_error_analysis = MLFFErrorAnalysis._from_data(mock_calculations)

    def rmse_error_force(mlff_force, dft_force, natoms):
        norm_error = np.linalg.norm(dft_force - mlff_force, axis=-1)
        rmse = np.sqrt(np.sum(norm_error**2, axis=-1) / (3 * natoms))
        return rmse

    expected_force_error = rmse_error_force(
        mlff_force=mlff_error_analysis.mlff.forces,
        dft_force=mlff_error_analysis.dft.forces,
        natoms=mlff_error_analysis.mlff.nions,
    )
    output_force_error = mlff_error_analysis.get_force_rmse()
    assert np.array_equal(expected_force_error, output_force_error)


def test_multiple_force_computation(mock_multiple_calculations):
    mlff_error_analysis = MLFFErrorAnalysis._from_data(mock_multiple_calculations)

    def rmse_error_force(mlff_force, dft_force, natoms, nconfig):
        norm_error = np.linalg.norm(dft_force - mlff_force, axis=-1)
        rmse = np.sqrt(np.sum(norm_error**2, axis=-1) / (3 * natoms))
        return np.sum(rmse, axis=-1) / nconfig

    expected_force_error = rmse_error_force(
        mlff_force=mlff_error_analysis.mlff.forces,
        dft_force=mlff_error_analysis.dft.forces,
        natoms=mlff_error_analysis.mlff.nions,
        nconfig=mlff_error_analysis.mlff.nconfig,
    )
    output_force_error = mlff_error_analysis.get_force_rmse(
        normalize_by_configurations=True
    )
    assert np.array_equal(expected_force_error, output_force_error)


def test_stress_error_computation(mock_calculations):
    mlff_error_analysis = MLFFErrorAnalysis._from_data(mock_calculations)

    def rmse_error_stress(mlff_stress, dft_stress, natoms):
        mlff_stress = np.triu(mlff_stress)
        dft_stress = np.triu(dft_stress)
        norm_error = np.linalg.norm(dft_stress - mlff_stress, axis=-1)
        rmse = np.sqrt(np.sum(norm_error**2, axis=-1) / 6)
        return rmse

    expected_stress_error = rmse_error_stress(
        mlff_stress=mlff_error_analysis.mlff.stresses,
        dft_stress=mlff_error_analysis.dft.stresses,
        natoms=mlff_error_analysis.mlff.nions,
    )
    output_stress_error = mlff_error_analysis.get_stress_rmse()
    assert np.array_equal(expected_stress_error, output_stress_error)


def test_multiple_stress_computation(mock_multiple_calculations):
    mlff_error_analysis = MLFFErrorAnalysis._from_data(mock_multiple_calculations)

    def rmse_error_stress(mlff_stress, dft_stress, nconfig):
        mlff_stress = np.triu(mlff_stress)
        dft_stress = np.triu(dft_stress)
        norm_error = np.linalg.norm(dft_stress - mlff_stress, axis=-1)
        rmse = np.sqrt(np.sum(norm_error**2, axis=-1) / 6)
        return np.sum(rmse, axis=-1) / nconfig

    expected_stress_error = rmse_error_stress(
        mlff_stress=mlff_error_analysis.mlff.stresses,
        dft_stress=mlff_error_analysis.dft.stresses,
        nconfig=mlff_error_analysis.mlff.nconfig,
    )
    output_stress_error = mlff_error_analysis.get_stress_rmse(
        normalize_by_configurations=True
    )
    assert np.array_equal(expected_stress_error, output_stress_error)
