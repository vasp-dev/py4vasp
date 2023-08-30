# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy
from typing import Sequence

import numpy as np
import pytest

from py4vasp._raw.data import CONTCAR, Cell, Structure, Topology
from py4vasp._raw.data_wrapper import VaspData
from py4vasp._util.parser import ParsePoscar
from py4vasp.exception import ParserError


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
        num_scaling_factors: int = 1,
        has_lattice: bool = True,
        has_species_name: bool = True,
        has_ion_per_species: bool = True,
        has_selective_dynamics: bool = False,
        has_ion_positions: bool = True,
        has_lattice_velocities: bool = False,
        has_ion_velocities: bool = False,
        coordinate_system: str = "Direct",
    ):
        comment_line = "Cubic BN" if has_comment_line else None
        scaling_factor = (
            [float(i + 1) for i in range(1, num_scaling_factors + 1)]
            if num_scaling_factors
            else None
        )
        lattice = (
            [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]] if has_lattice else None
        )
        species_names = ["B", "N"] if has_species_name else None
        ions_per_species = [1, 1] if has_ion_per_species else None
        selective_dynamics = "Selective dynamics" if has_selective_dynamics else None
        positions = [[0.0, 0.0, 0.0], [0.25, 0.0, 0.25]]
        if coordinate_system == "Direct":
            ion_positions = [["Direct"]] + positions if has_ion_positions else None
            ion_positions_direct = copy.deepcopy(ion_positions)
        elif coordinate_system == "Cartesian":
            if positions is None:
                ion_positions_direct = None
                ion_positions = None
            else:
                ion_positions_direct = [["Direct"]] + positions
                positions = np.array(positions)
                lattice = np.array(lattice)
                positions = positions @ lattice.T
                lattice = lattice.tolist()
                ion_positions = [["Cartesian"]] + positions.tolist()
        if has_selective_dynamics:
            ion_positions[1].append("T F T")
            ion_positions[2].append("F T F")
            ion_positions_direct[1].append("T F T")
            ion_positions_direct[2].append("F T F")
        if num_scaling_factors > 0:
            scaled_lattice = np.array(lattice) * np.array(scaling_factor).T
            scaled_lattice = scaled_lattice.tolist()
        else:
            scaled_lattice = lattice
        _lattice_velocities = [[0.0, -0.6, 0.2], [0.1, 0.3, -0.2], [0.2, -0.4, 0.4]]
        lattice_velocities = (
            [["Lattice velocities and vectors"], [1]]
            + _lattice_velocities
            + scaled_lattice
            if has_lattice_velocities
            else None
        )
        ion_velocities = (
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] if has_ion_velocities else None
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
        poscar_input_string = poscar_creator(*componentwise_input)
        expected_output = [
            comment_line,
            scaling_factor,
            lattice,
            species_names,
            ions_per_species,
            selective_dynamics,
            ion_positions_direct,
            lattice_velocities,
            ion_velocities,
        ]
        arguments = {}
        if not has_species_name:
            arguments["species_name"] = "B N"
        return poscar_input_string, expected_output, arguments

    return _cubic_BN


def test_cubic_BN_fixture_defaults(cubic_BN):
    output_poscar_string, *_ = cubic_BN()
    expected_poscar_string = """Cubic BN
2.0
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
B N
1 1
Direct
0.0 0.0 0.0
0.25 0.0 0.25"""
    assert output_poscar_string == expected_poscar_string


def test_cubic_BN_fixture_scaling_factor(cubic_BN):
    output_poscar_string, *_ = cubic_BN(num_scaling_factors=3)
    expected_poscar_string = """Cubic BN
2.0 3.0 4.0
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
B N
1 1
Direct
0.0 0.0 0.0
0.25 0.0 0.25"""
    assert output_poscar_string == expected_poscar_string


def test_cubic_BN_fixture_species_name_provided(cubic_BN):
    output_poscar_string, *_ = cubic_BN(has_species_name=False)
    expected_poscar_string = """Cubic BN
2.0
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
1 1
Direct
0.0 0.0 0.0
0.25 0.0 0.25"""
    assert output_poscar_string == expected_poscar_string


def test_cubic_BN_fixture_selective_dynamics(cubic_BN):
    output_poscar_string, *_ = cubic_BN(has_selective_dynamics=True)
    expected_poscar_string = """Cubic BN
2.0
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
B N
1 1
Selective dynamics
Direct
0.0 0.0 0.0 T F T
0.25 0.0 0.25 F T F"""
    assert output_poscar_string == expected_poscar_string


def test_cubic_BN_fixture_cartesian(cubic_BN):
    output_poscar_string, *_ = cubic_BN(coordinate_system="Cartesian")
    expected_poscar_string = """Cubic BN
2.0
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
B N
1 1
Cartesian
0.0 0.0 0.0
0.125 0.25 0.125"""
    assert output_poscar_string == expected_poscar_string


def test_cubic_BN_fixture_lattice_velocities(cubic_BN):
    output_poscar_string, *_ = cubic_BN(has_lattice_velocities=True)
    expected_poscar_string = """Cubic BN
2.0
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
B N
1 1
Direct
0.0 0.0 0.0
0.25 0.0 0.25
Lattice velocities and vectors
1
0.0 -0.6 0.2
0.1 0.3 -0.2
0.2 -0.4 0.4
0.0 1.0 1.0
1.0 0.0 1.0
1.0 1.0 0.0"""
    assert output_poscar_string == expected_poscar_string


def test_comment_line(cubic_BN):
    poscar_string, componentwise_inputs, _ = cubic_BN()
    comment_line = componentwise_inputs[0]
    parsed_comment_line = ParsePoscar(poscar_string).comment_line
    assert comment_line == parsed_comment_line


def test_error_no_scaling_factor_provided(cubic_BN):
    poscar_string, *_ = cubic_BN(num_scaling_factors=0)
    with pytest.raises(ParserError):
        ParsePoscar(poscar_string).cell


@pytest.mark.parametrize("num_scaling_factors", [1, 3])
def test_cell(cubic_BN, num_scaling_factors, Assert):
    poscar_string, componentwise_inputs, _ = cubic_BN(
        num_scaling_factors=num_scaling_factors
    )
    scaling_factor = componentwise_inputs[1]
    unscaled_lattice = componentwise_inputs[2]
    # Performed in the convention of how VASP manages the scaling factor
    # If scaling factor is a float, then it is preserved from POSCAR -> CONTCAR
    # However, if it is a Sequence, then is preserves only the scaled lattice vectors
    # and the scaling factor is set to 1.0. If a negative number is set, then it
    # it is interpreted as the desired volume of the cell and the scaling factor
    # is computed accordingly (see next test).
    if isinstance(scaling_factor, float) or len(scaling_factor) == 1:
        _scaling_factor = (
            scaling_factor if isinstance(scaling_factor, float) else scaling_factor[0]
        )
        expected_cell = Cell(lattice_vectors=unscaled_lattice, scale=_scaling_factor)
    else:
        scaled_lattice = np.array(unscaled_lattice) * scaling_factor
        expected_cell = Cell(lattice_vectors=scaled_lattice, scale=1.0)
    output_cell = ParsePoscar(poscar_string).cell
    Assert.allclose(expected_cell.lattice_vectors, output_cell.lattice_vectors)
    Assert.allclose(expected_cell.scale, output_cell.scale)


def test_negative_scaling_factor(cubic_BN, poscar_creator, Assert):
    _, componentwise_inputs, _ = cubic_BN(num_scaling_factors=1)
    scaling_factor = -27  # Negative number interpreted as expected volume
    componentwise_inputs[1] = scaling_factor
    poscar_string = poscar_creator(*componentwise_inputs)
    unscaled_lattice = componentwise_inputs[2]
    volume_of_cell = np.dot(
        unscaled_lattice[0], np.cross(unscaled_lattice[1], unscaled_lattice[2])
    )
    expected_scaling_factor = (-scaling_factor / volume_of_cell) ** (1 / 3)
    expected_cell = Cell(
        lattice_vectors=unscaled_lattice, scale=expected_scaling_factor
    )
    output_cell = ParsePoscar(poscar_string).cell
    Assert.allclose(expected_cell.lattice_vectors, output_cell.lattice_vectors)
    Assert.allclose(expected_cell.scale, output_cell.scale)


@pytest.mark.parametrize("has_species_name", [True, False])
def test_topology(cubic_BN, has_species_name, Assert):
    poscar_string, componentwise_inputs, arguments = cubic_BN(
        has_species_name=has_species_name
    )
    species_names = componentwise_inputs[3]
    ions_per_species = componentwise_inputs[4]
    expected_species_names = (
        VaspData(species_names) if species_names else arguments["species_name"].split()
    )
    expected_ions_per_species = VaspData(ions_per_species)
    expected_topology = Topology(
        number_ion_types=expected_ions_per_species, ion_types=expected_species_names
    )
    output_topology = ParsePoscar(poscar_string, **arguments).topology
    Assert.allclose(
        expected_topology.number_ion_types, output_topology.number_ion_types
    )
    assert np.all(expected_topology.ion_types == output_topology.ion_types)


def test_error_no_species_provided(cubic_BN):
    poscar_string, *_ = cubic_BN(has_species_name=False)
    with pytest.raises(ParserError):
        ParsePoscar(poscar_string).topology


@pytest.mark.parametrize("has_selective_dynamics", [True, False])
def test_has_selective_dynamics(cubic_BN, has_selective_dynamics):
    poscar_string, *_ = cubic_BN(has_selective_dynamics=has_selective_dynamics)
    output_has_selective_dynamics = ParsePoscar(poscar_string).has_selective_dynamics
    assert has_selective_dynamics == output_has_selective_dynamics


@pytest.mark.parametrize("has_species_name", [True, False])
@pytest.mark.parametrize("has_selective_dynamics", [True, False])
def test_positions_direct(cubic_BN, has_species_name, has_selective_dynamics, Assert):
    poscar_string, componentwise_inputs, arguments = cubic_BN(
        has_species_name=has_species_name, has_selective_dynamics=has_selective_dynamics
    )
    ion_positions = componentwise_inputs[6]
    assert ion_positions[0][0] == "Direct"
    expected_ion_positions = [x[0:3] for x in ion_positions[1:]]
    if has_selective_dynamics:
        expected_selective_dynamics = [x[3:] for x in ion_positions[1:]]
        expected_selective_dynamics = [
            item for sublist in expected_selective_dynamics for item in sublist
        ]
        expected_selective_dynamics = [x.split() for x in expected_selective_dynamics]
        expected_selective_dynamics = [
            [True if x == "T" else False for x in sublist]
            for sublist in expected_selective_dynamics
        ]
    else:
        expected_selective_dynamics = False
    expected_ion_positions = VaspData(expected_ion_positions)
    expected_selective_dynamics = VaspData(expected_selective_dynamics)
    output_ion_positions, output_selective_dynamics = ParsePoscar(
        poscar_string, **arguments
    ).ion_positions_and_selective_dynamics
    Assert.allclose(expected_ion_positions, output_ion_positions)
    Assert.allclose(expected_selective_dynamics, output_selective_dynamics)


@pytest.mark.parametrize("has_species_name", [True, False])
@pytest.mark.parametrize("has_selective_dynamics", [True, False])
@pytest.mark.parametrize("num_scaling_factors", [1, 3])
def test_positions_cartesian(
    cubic_BN, has_species_name, has_selective_dynamics, num_scaling_factors
):
    poscar_string, componentwise_inputs, arguments = cubic_BN(
        has_species_name=has_species_name,
        has_selective_dynamics=has_selective_dynamics,
        coordinate_system="Cartesian",
        num_scaling_factors=num_scaling_factors,
    )
    ion_positions = componentwise_inputs[6]
    expected_ion_positions = [x[0:3] for x in ion_positions[1:]]
    if has_selective_dynamics:
        expected_selective_dynamics = [x[3:] for x in ion_positions[1:]]
        expected_selective_dynamics = [
            item for sublist in expected_selective_dynamics for item in sublist
        ]
        expected_selective_dynamics = [x.split() for x in expected_selective_dynamics]
        expected_selective_dynamics = [
            [True if x == "T" else False for x in sublist]
            for sublist in expected_selective_dynamics
        ]
    else:
        expected_selective_dynamics = False
    expected_ion_positions = VaspData(expected_ion_positions)
    expected_selective_dynamics = VaspData(expected_selective_dynamics)
    output_ion_positions, output_selective_dynamics = ParsePoscar(
        poscar_string, **arguments
    ).ion_positions_and_selective_dynamics
    assert np.allclose(expected_ion_positions, output_ion_positions)
    assert np.allclose(expected_selective_dynamics, output_selective_dynamics)


def test_lattice_velocities(cubic_BN, Assert):
    poscar_string, componentwise_inputs, arguments = cubic_BN(
        has_lattice_velocities=True
    )
    lattice_velocities = componentwise_inputs[7]
    assert lattice_velocities[0][0] == "Lattice velocities and vectors"
    expected_lattice_velocities = [x[0:3] for x in lattice_velocities[2:5]]
    expected_lattice_velocities = VaspData(expected_lattice_velocities)
    output_lattice_velocities = ParsePoscar(
        poscar_string, **arguments
    ).lattice_velocities
    Assert.allclose(expected_lattice_velocities, output_lattice_velocities)


@pytest.mark.parametrize("has_species_name", [True, False])
def test_to_contcar(cubic_BN, has_species_name, Assert):
    poscar_string, componentwise_inputs, arguments = cubic_BN(
        has_species_name=has_species_name
    )
    output_contcar = ParsePoscar(poscar_string, **arguments).to_contcar()
    assert isinstance(output_contcar, CONTCAR)
    assert isinstance(output_contcar.structure, Structure)
    assert isinstance(output_contcar.selective_dynamics, VaspData)
    assert isinstance(output_contcar.lattice_velocities, VaspData)
