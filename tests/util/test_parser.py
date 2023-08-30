# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from typing import Sequence

import numpy as np
import pytest

from py4vasp._raw.data import CONTCAR, Cell, Structure, Topology
from py4vasp._raw.data_wrapper import VaspData
from py4vasp._util.parser import ParsePoscar


@pytest.fixture
def poscar_creator():
    def _poscar_creator(
        comment_line: str or None,
        scaling_factor: Sequence[float] or float or None,
        lattice: Sequence[float] or None,
        species_names: Sequence[str] or None,
        ions_per_species: Sequence[int],
        selective_dynamics: str or None,
        ion_positions: Sequence[float] or None,
        lattice_velocities: Sequence[float] or None,
        ion_velocities: Sequence[float] or None,
    ) -> str:
        if isinstance(scaling_factor, (int, float)):
            scaling_factor = [scaling_factor]
        input_comment_line = comment_line if comment_line else None
        input_scaling_factor = (
            " ".join([str(x) for x in scaling_factor]) if scaling_factor else None
        )
        input_lattice = (
            "\n".join([" ".join([str(y) for y in x]) for x in lattice])
            if lattice
            else None
        )
        input_species_names = " ".join(species_names) if species_names else None
        input_ions_per_species = (
            " ".join([str(x) for x in ions_per_species]) if ions_per_species else None
        )
        input_selective_dynamics = selective_dynamics if selective_dynamics else None
        input_ion_positions = (
            "\n".join([" ".join([str(y) for y in x]) for x in ion_positions])
            if ion_positions
            else None
        )
        input_lattice_velocities = (
            "\n".join([" ".join([str(y) for y in x]) for x in lattice_velocities])
            if lattice_velocities
            else None
        )
        input_ion_velocities = (
            "\n".join([" ".join([str(y) for y in x]) for x in ion_velocities])
            if ion_velocities
            else None
        )
        componentwise_input = [
            input_comment_line,
            input_scaling_factor,
            input_lattice,
            input_species_names,
            input_ions_per_species,
            input_selective_dynamics,
            input_ion_positions,
            input_lattice_velocities,
            input_ion_velocities,
        ]
        poscar_input_string = "\n".join(filter(None, componentwise_input))
        return poscar_input_string

    return _poscar_creator


@pytest.fixture
def cubic_BN(poscar_creator):
    def _cubic_BN(
        has_comment_line: bool = True,
        num_scaling_factors: int = True,
        has_lattice: bool = True,
        has_species_name: bool = True,
        has_ion_per_species: bool = True,
        has_selective_dynamics: bool = False,
        has_ion_positions: bool = True,
        has_lattice_velocities: bool = False,
        has_ion_velocities: bool = False,
    ):
        comment_line = "Cubic BN" if has_comment_line else None
        scaling_factor = (
            [float(i) for i in range(1, num_scaling_factors + 1)]
            if num_scaling_factors
            else None
        )
        lattice = (
            [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]] if has_lattice else None
        )
        species_names = ["B", "N"] if has_species_name else None
        ions_per_species = [1, 1] if has_ion_per_species else None
        selective_dynamics = "Selective dynamics" if has_selective_dynamics else None
        ion_positions = (
            [["Direct"], [0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
            if has_ion_positions
            else None
        )
        lattice_velocities = (
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] if has_lattice_velocities else None
        )
        ion_velocities = (
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] if has_ion_velocities else None
        )
        poscar_input_string = poscar_creator(
            comment_line,
            scaling_factor,
            lattice,
            species_names,
            ions_per_species,
            selective_dynamics,
            ion_positions,
            lattice_velocities,
            ion_velocities,
        )
        componentwise_input = [
            comment_line,
            scaling_factor,
            lattice,
            species_names,
            ions_per_species,
            selective_dynamics,
            ion_positions,
            lattice_velocities,
            ion_velocities,
        ]
        arguments = {}
        if not has_species_name:
            arguments["species_name"] = ["B", "N"]
        return poscar_input_string, componentwise_input, arguments

    return _cubic_BN


def test_cubic_BN_fixture_defaults(cubic_BN):
    output_poscar_string, *_ = cubic_BN()
    expected_poscar_string = """Cubic BN
1.0
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
B N
1 1
Direct
0.0 0.0 0.0
0.25 0.25 0.25"""
    assert output_poscar_string == expected_poscar_string


def test_cubic_BN_fixture_scaling_factor(cubic_BN):
    output_poscar_string, *_ = cubic_BN(num_scaling_factors=3)
    expected_poscar_string = """Cubic BN
1.0 2.0 3.0
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
B N
1 1
Direct
0.0 0.0 0.0
0.25 0.25 0.25"""
    assert output_poscar_string == expected_poscar_string


def test_cubic_BN_fixture_species_name_provided(cubic_BN):
    output_poscar_string, *_ = cubic_BN(has_species_name=False)
    expected_poscar_string = """Cubic BN
1.0
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
1 1
Direct
0.0 0.0 0.0
0.25 0.25 0.25"""
    assert output_poscar_string == expected_poscar_string


def test_comment_line(cubic_BN):
    poscar_string, componentwise_inputs, _ = cubic_BN()
    comment_line = componentwise_inputs[0]
    parsed_comment_line = ParsePoscar(poscar_string).comment_line
    assert comment_line == parsed_comment_line


@pytest.mark.parametrize("num_scaling_factors", [0, 1, 3])
def test_cell(cubic_BN, _scaling_factor, Assert):
    poscar_string, componentwise_inputs, _ = cubic_BN()
    _scaling_factor = componentwise_inputs[1]
    _lattice = componentwise_inputs[2]
    lattice_vectors = np.array([x.split() for x in _lattice.split("\n")], dtype=float)
    scaling_factor = np.array(_scaling_factor.split(), dtype=float)
    if len(scaling_factor) == 1:
        scaling_factor = scaling_factor[0]
    lattice_vectors = VaspData(lattice_vectors)
    expected_cell = Cell(lattice_vectors=lattice_vectors, scale=scaling_factor)
    output_cell = ParsePoscar(poscar_string).cell
    Assert.allclose(expected_cell.lattice_vectors, output_cell.lattice_vectors)
    Assert.allclose(expected_cell.scale, output_cell.scale)


@pytest.mark.parametrize("species_name_provided", [True, False])
def test_topology(cubic_BN, species_name_provided, Assert):
    poscar_string, componentwise_inputs, arguments = cubic_BN(
        species_name_provided=species_name_provided
    )
    if species_name_provided:
        _species_names = componentwise_inputs[3]
        _ions_per_species = componentwise_inputs[4]
    else:
        _species_names = arguments[0]
        _ions_per_species = componentwise_inputs[3]
    expected_species_names = _species_names.split()
    expected_species_names = np.array(expected_species_names)
    expected_ions_per_species = np.array(_ions_per_species.split(), dtype=int)
    expected_species_names = VaspData(expected_species_names)
    expected_ions_per_species = VaspData(expected_ions_per_species)
    expected_topology = Topology(
        number_ion_types=expected_ions_per_species, ion_types=expected_species_names
    )
    if not arguments:
        output_topology = ParsePoscar(poscar_string).topology
    else:
        output_topology = ParsePoscar(poscar_string, *arguments).topology
    Assert.allclose(
        expected_topology.number_ion_types, output_topology.number_ion_types
    )
    assert np.all(expected_topology.ion_types == output_topology.ion_types)


@pytest.mark.parametrize("species_name_provided", [True, False])
def test_positions_direct(cubic_BN, species_name_provided, Assert):
    poscar_string, componentwise_inputs, arguments = cubic_BN(
        species_name_provided=species_name_provided
    )
    if species_name_provided:
        _ion_positions = componentwise_inputs[5]
    else:
        _ion_positions = componentwise_inputs[4]
    ion_positions = [x.split() for x in _ion_positions.split("\n")]
    assert ion_positions[0][0] == "Direct"
    expected_ion_positions = np.array(ion_positions[1:], dtype=float)
    expected_ion_positions = VaspData(expected_ion_positions)
    if not arguments:
        output_ion_positions = ParsePoscar(poscar_string).ion_positions
    else:
        output_ion_positions = ParsePoscar(poscar_string, *arguments).ion_positions
    Assert.allclose(expected_ion_positions, output_ion_positions)


@pytest.mark.parametrize("species_name_provided", [True, False])
def test_to_contcar(cubic_BN, species_name_provided, Assert):
    poscar_string, componentwise_inputs, arguments = cubic_BN(
        species_name_provided=species_name_provided
    )
    if not arguments:
        output_contcar = ParsePoscar(poscar_string).to_contcar()
    else:
        output_contcar = ParsePoscar(poscar_string, *arguments).to_contcar()
    assert isinstance(output_contcar, CONTCAR)
    assert isinstance(output_contcar.structure, Structure)
