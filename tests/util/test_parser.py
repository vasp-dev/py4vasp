# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy
from typing import List, Sequence

import numpy as np
import pytest

from py4vasp._raw.data import CONTCAR, Cell, Structure, Stoichiometry
from py4vasp._raw.data_wrapper import VaspData
from py4vasp._util.parser import ParsePoscar
from py4vasp.exception import ParserError


@pytest.fixture
def poscar_creator():
    def _to_string(input: Sequence[int or float] or int or float or None) -> str:
        if input is None:
            return None
        if not isinstance(input, Sequence):
            return " ".join([str(input)])
        if not all(isinstance(x, list) for x in input):
            return " ".join([str(x) for x in input])
        else:
            return "\n".join([" ".join([str(y) for y in x]) for x in input])

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
        scaling_factor = _to_string(scaling_factor)
        lattice = _to_string(lattice)
        species_names = _to_string(species_names)
        ions_per_species = _to_string(ions_per_species)
        ion_positions = _to_string(ion_positions)
        lattice_velocities = _to_string(lattice_velocities)
        ion_velocities = _to_string(ion_velocities)
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
        poscar_input_string = "\n".join(filter(None, componentwise_input))
        return poscar_input_string

    return _poscar_creator


@pytest.fixture
def cubic_BN(poscar_creator):
    def _assign_value(value, has_value: bool or None):
        if has_value:
            return value
        else:
            return None

    def _add_header(header: str, contents: List):
        if not contents:
            return None
        contents = [[header]] + contents
        return contents

    def _convert_coordinate_system(
        from_system: str, to_system: str, quantity: List, lattice: List or None
    ):
        if not lattice:
            return None
        if from_system == "Direct" and to_system == "Cartesian":
            quantity = np.array(quantity) @ np.array(lattice).T
            quantity = np.round(quantity, 8)
            quantity_new_sys = _add_header(
                header="Cartesian", contents=quantity.tolist()
            )
        return quantity_new_sys

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
        ions_coordinate_system: str = "Direct",
        velocity_coordinate_system: str = "Cartesian",
    ):
        DEFAULTS = {
            "comment_line": "Cubic BN",
            "scaling_factor": [float(i + 1) for i in range(1, num_scaling_factors + 1)],
            "lattice": [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
            "species_names": ["B", "N"],
            "ions_per_species": [1, 1],
            "selective_dynamics": "Selective dynamics",
            "ion_positions": [[0.0, 0.0, 0.0], [0.25, 0.0, 0.25]],
            "lattice_velocities": [
                [1],
                [0.0, -0.6, 0.2],
                [0.1, 0.3, -0.2],
                [0.2, -0.4, 0.4],
            ],
            "ion_velocities": [[0.2, 0.4, -0.2], [0.4, 0.6, -0.3]],
            "ions_coordinate_system": "Direct",
            "velocity_coordinate_system": "Direct",
        }
        comment_line = DEFAULTS["comment_line"]
        comment_line = _assign_value(comment_line, has_comment_line)
        scaling_factor = DEFAULTS["scaling_factor"]
        scaling_factor = _assign_value(scaling_factor, num_scaling_factors > 0)
        lattice = DEFAULTS["lattice"]
        lattice = _assign_value(lattice, has_lattice)
        species_names = DEFAULTS["species_names"]
        species_names = _assign_value(species_names, has_species_name)
        ions_per_species = DEFAULTS["ions_per_species"]
        ions_per_species = _assign_value(ions_per_species, has_ion_per_species)
        selective_dynamics = DEFAULTS["selective_dynamics"]
        selective_dynamics = _assign_value(selective_dynamics, has_selective_dynamics)
        positions_writeout = _add_header(
            header=DEFAULTS["ions_coordinate_system"],
            contents=DEFAULTS["ion_positions"],
        )
        ion_positions_direct = _assign_value(positions_writeout, has_ion_positions)
        if ions_coordinate_system == DEFAULTS["ions_coordinate_system"]:
            ion_positions = copy.deepcopy(ion_positions_direct)
        else:
            positions_writeout = _convert_coordinate_system(
                from_system=DEFAULTS["ions_coordinate_system"],
                to_system=ions_coordinate_system,
                quantity=DEFAULTS["ion_positions"],
                lattice=DEFAULTS["lattice"],
            )
            ion_positions = _assign_value(positions_writeout, has_ion_positions)
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
        _lattice_velocities = []
        _lattice_velocities.extend(DEFAULTS["lattice_velocities"])
        _lattice_velocities.extend(scaled_lattice)
        lattice_velocities = _assign_value(_lattice_velocities, has_lattice_velocities)
        lattice_velocities = _add_header(
            header="Lattice velocities and vectors", contents=_lattice_velocities
        )
        lattice_velocities = _assign_value(lattice_velocities, has_lattice_velocities)
        if velocity_coordinate_system == DEFAULTS["velocity_coordinate_system"]:
            ion_velocities = _add_header(
                header=DEFAULTS["velocity_coordinate_system"],
                contents=DEFAULTS["ion_velocities"],
            )
        else:
            ion_velocities = _convert_coordinate_system(
                from_system=DEFAULTS["velocity_coordinate_system"],
                to_system=velocity_coordinate_system,
                quantity=DEFAULTS["ion_velocities"],
                lattice=DEFAULTS["lattice"],
            )
        ion_velocities = _assign_value(ion_velocities, has_ion_velocities)
        if velocity_coordinate_system == "Cartesian":
            output_ion_velocities = ion_velocities
        else:
            output_ion_velocities = _convert_coordinate_system(
                from_system=velocity_coordinate_system,
                to_system="Cartesian",
                quantity=ion_velocities[1:],
                lattice=lattice,
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
            output_ion_velocities,
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
    output_poscar_string, *_ = cubic_BN(ions_coordinate_system="Cartesian")
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


def test_cubic_BN_fixture_ion_velocities(cubic_BN):
    output_poscar_string, *_ = cubic_BN(
        has_lattice_velocities=True,
        has_ion_velocities=True,
        velocity_coordinate_system="Cartesian",
    )
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
1.0 1.0 0.0
Cartesian
0.1 0.0 0.3
0.15 0.05 0.5"""
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
def test_stoichiometry(cubic_BN, has_species_name, Assert):
    poscar_string, componentwise_inputs, arguments = cubic_BN(
        has_species_name=has_species_name
    )
    species_names = componentwise_inputs[3]
    ions_per_species = componentwise_inputs[4]
    expected_species_names = (
        VaspData(species_names) if species_names else arguments["species_name"].split()
    )
    expected_ions_per_species = VaspData(ions_per_species)
    expected_stoichiometry = Stoichiometry(
        number_ion_types=expected_ions_per_species, ion_types=expected_species_names
    )
    output_stoichiometry = ParsePoscar(poscar_string, **arguments).stoichiometry
    Assert.allclose(
        expected_stoichiometry.number_ion_types, output_stoichiometry.number_ion_types
    )
    if has_species_name:
        expected_ion_types = expected_stoichiometry.ion_types.__array__()
    else:
        expected_ion_types = expected_stoichiometry.ion_types
    output_ion_types = output_stoichiometry.ion_types.__array__()
    assert all(expected_ion_types == output_ion_types)


def test_error_no_species_provided(cubic_BN):
    poscar_string, *_ = cubic_BN(has_species_name=False)
    with pytest.raises(ParserError):
        ParsePoscar(poscar_string).stoichiometry


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
        ions_coordinate_system="Cartesian",
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


def test_no_lattice_velocities(cubic_BN):
    poscar_string, *_ = cubic_BN(has_lattice_velocities=False)
    with pytest.raises(ParserError):
        ParsePoscar(poscar_string).lattice_velocities


@pytest.mark.parametrize("velocity_coordinate_system", ["Cartesian", "Direct"])
def test_ion_velocities(cubic_BN, velocity_coordinate_system):
    poscar_string, componentwise_inputs, arguments = cubic_BN(
        has_lattice_velocities=True,
        has_ion_velocities=True,
        velocity_coordinate_system=velocity_coordinate_system,
    )
    ion_velocities = componentwise_inputs[8]
    expected_ion_velocities = [x[0:3] for x in ion_velocities[1:]]
    expected_ion_velocities = VaspData(expected_ion_velocities)
    output_ion_velocities = ParsePoscar(poscar_string, **arguments).ion_velocities
    assert np.allclose(expected_ion_velocities, output_ion_velocities)


@pytest.mark.parametrize("has_species_name", [True, False])
def test_to_contcar(cubic_BN, has_species_name, Assert):
    poscar_string, componentwise_inputs, arguments = cubic_BN(
        has_species_name=has_species_name,
        has_selective_dynamics=True,
        has_lattice_velocities=True,
        has_ion_velocities=True,
    )
    output_contcar = ParsePoscar(poscar_string, **arguments).to_contcar()
    assert isinstance(output_contcar, CONTCAR)
    assert isinstance(output_contcar.structure, Structure)
    assert isinstance(output_contcar.selective_dynamics, VaspData)
    assert isinstance(output_contcar.lattice_velocities, VaspData)
