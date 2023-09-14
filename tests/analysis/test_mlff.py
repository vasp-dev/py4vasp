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

from py4vasp import Calculation
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
    assert isinstance(error_analysis.mlff_nions, int)
    assert isinstance(error_analysis.dft_nions, int)
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
    assert isinstance(error_analysis.mlff_nions, int)
    assert isinstance(error_analysis.dft_nions, int)
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
