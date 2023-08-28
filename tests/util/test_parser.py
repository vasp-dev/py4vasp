import numpy as np
import pytest

from py4vasp._raw.data import Cell, Topology
from py4vasp._raw.data_wrapper import VaspData
from py4vasp._util.parser import ParsePoscar


@pytest.fixture
def cubic_BN():
    def _cubic_BN(scaling_factor=3.57, species_name_provided=True):
        if isinstance(scaling_factor, float) or isinstance(scaling_factor, int):
            scaling_factor = [scaling_factor]
        _comment_line = "Cubic BN"
        _scaling_factor = " ".join([str(x) for x in scaling_factor])
        _lattice = """0.0 0.5 0.5\n0.5 0.0 0.5\n0.5 0.5 0.0"""
        _species_names = "B N"
        _ions_per_species = "1 1"
        _ion_positions = """Direct\n0.0 0.0 0.0\n0.25 0.25 0.25"""
        if species_name_provided:
            componentwise_inputs = [
                _comment_line,
                _scaling_factor,
                _lattice,
                _species_names,
                _ions_per_species,
                _ion_positions,
            ]
            arguments = None
        else:
            componentwise_inputs = [
                _comment_line,
                _scaling_factor,
                _lattice,
                _ions_per_species,
                _ion_positions,
            ]
            arguments = [_species_names]

        poscar_input_string = "\n".join(componentwise_inputs)
        return poscar_input_string, componentwise_inputs, arguments

    return _cubic_BN


def test_cubic_BN_fixture_defaults(cubic_BN):
    output_poscar_string, *_ = cubic_BN()
    expected_poscar_string = """Cubic BN
3.57
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
    output_poscar_string, *_ = cubic_BN(scaling_factor=[3.57, 3.57, 3.57])
    expected_poscar_string = """Cubic BN
3.57 3.57 3.57
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
    output_poscar_string, *_ = cubic_BN(species_name_provided=False)
    expected_poscar_string = """Cubic BN
3.57
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


@pytest.mark.parametrize("_scaling_factor", [3.57, [3.57, 3.57, 3.57]])
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
